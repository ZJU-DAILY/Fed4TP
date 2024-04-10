from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, merge_param_dict
from federatedscope.core.workers.base_server import BaseServer
import logging
import copy
import os
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Server(BaseServer):
    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5,
                 total_round_num=10, device='cpu', strategy=None, **kwargs):
        super(Server, self).__init__(ID, state, config, model, strategy)
        self._register_default_handlers()
        self.data = data
        self.device = device
        self.best_results = dict()
        self.history_results = dict()
        self.aggregator = get_aggregator(self._cfg.federate.method, model=model, device=device, config=self._cfg)
        self.model_num = config.model.model_num_per_trainer
        self.client_num = client_num
        self.total_round_num = total_round_num
        self.sample_client_num = int(self._cfg.federate.sample_client_num)
        self.join_in_client_num = 0
        self.join_in_info = dict()
        self.is_finish = False
        self.sampler = get_sampler(sample_strategy=self._cfg.federate.sampler, client_num=self.client_num)
        self.msg_buffer = {'train': dict(), 'eval': dict()}
        comm_queue = kwargs.get('shared_comm_queue', None)
        if self._cfg.federate.process_num > 1:
            id2comm = kwargs.get('id2comm', None)
            self.comm_manager = StandaloneDDPCommManager(comm_queue=comm_queue, monitor=self._monitor,
                                                         id2comm=id2comm)
        else:
            self.comm_manager = StandaloneCommManager(comm_queue=comm_queue, monitor=self._monitor)

        if self._cfg.fedtfp.mg_use:
            self.model_num_fedtfp = 2
            self.clusters_fedtfp = [[1, 2, 3],
                                    [4, 5, 6]]  # clients clustering results can be gotten from cluster folder.
            self.global_aggregator_fedtfp = get_aggregator(self._cfg.federate.method, model=model, device=device,
                                                           config=self._cfg)
            self.aggregators_fedtfp = [get_aggregator(
                self._cfg.federate.method, model=model, device=device, config=self._cfg)
                for _ in range(self.model_num_fedtfp)]
            self.models_fedtfp = [copy.deepcopy(self.model) for _ in range(self.model_num_fedtfp)]
        if self._cfg.fedtfp.tm_use:
            self.time_window = 0

    def check_and_move_on(self, check_eval_result=False, min_received_num=None):
        if min_received_num is None:
            min_received_num = self._cfg.federate.sample_client_num
        if check_eval_result and self._cfg.federate.mode.lower() == "standalone":
            min_received_num = len(self.comm_manager.get_neighbors().keys())
        move_on_flag = True
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                if self._cfg.fedtfp.use:
                    unreliable_clients = self._perform_federated_aggregation()
                else:
                    self._perform_federated_aggregation()
                self.state += 1
                if not self._cfg.fedtfp.mg_use and self.state != self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end of round {self.state - 1}.')
                    self.broadcast_model_para(msg_type='evaluate')
                if self._cfg.fedtfp.tm_use:
                    if self.state == self.total_round_num and self.time_window < self._cfg.fedtfp.time_window_num - 1:
                        self.time_window += 1
                        self.state = 0
                if self.state < self.total_round_num:
                    logger.info(f'----------- Starting a new training round (Round #{self.state}) -------------')
                    self.msg_buffer['train'][self.state] = dict()
                    # Start a new training round
                    if self._cfg.fedtfp.mg_use:
                        if self._cfg.fedtfp.use:
                            self.broadcast_model_para(unreliable_clients)
                        else:
                            self.broadcast_model_para()
                    else:
                        if self._cfg.fedtfp.use:
                            self.broadcast_model_para(unreliable_clients)
                        else:
                            self.broadcast_model_para(msg_type='model_para')
                else:
                    logger.info('Server: Training is finished! Starting evaluation.')
                    self.broadcast_model_para(msg_type='evaluate')
            else:
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True
        else:
            move_on_flag = False
        return move_on_flag

    def _perform_federated_aggregation(self):
        if self._cfg.fedtfp.gd_use:
            unreliable_clients = [[0, 0, 0], [0, 0, 0]]
            if self.state >= self.last_gradients_length:
                gradient_list = list()
                for i in range(self.client_num):
                    for j in range(self.last_gradients_length):
                        gradient_list.append(
                            torch.cat([para.flatten(0) for para in self.last_gradients_lists[i][j].values()]))
                    self.last_gradients_lists[i].pop(0)
                median_gradient = torch.median(torch.stack(gradient_list), dim=0).values

        if self._cfg.fedtfp.mg_use:
            train_msg_buffer = self.msg_buffer['train'][self.state]
            msg_lists = [list() for _ in range(self.model_num_fedtfp)]
            for client_id in train_msg_buffer.keys():
                for i in range(self.model_num_fedtfp):
                    if client_id in self.clusters_fedtfp[i]:
                        train_data_size, gradient = train_msg_buffer[client_id]
                        msg_lists[i].append((train_data_size, gradient))
                        if self._cfg.fedtfp.gd_use:
                            if self.state >= self.last_gradients_length:
                                similarity = F.cosine_similarity(median_gradient,
                                                                 torch.cat(
                                                                     [para.flatten(0) for para in gradient.values()]),
                                                                 dim=0).item()
                                if similarity < self.unreliable_gradients_threshold:
                                    unreliable_clients[i][client_id - i * self.model_num_fedtfp] = 1
                            self.last_gradients_lists[client_id - 1].append(gradient)
                        break

            msg_list = list()
            for model_idx in range(self.model_num_fedtfp):
                agg_info = {'client_feedback': msg_lists[model_idx], 'recover_fun': None}
                msg_list.append((1, self.aggregators_fedtfp[model_idx].aggregate(agg_info)))

            agg_info = {'client_feedback': msg_list, 'recover_fun': None}
            result = self.global_aggregator_fedtfp.aggregate(agg_info)
            for model_idx in range(self.model_num_fedtfp):
                merged_param = merge_param_dict(self.models_fedtfp[model_idx].state_dict().copy(), result)
                self.models_fedtfp[model_idx].load_state_dict(merged_param, strict=False)
        else:
            train_msg_buffer = self.msg_buffer['train'][self.state]
            msg_list = list()
            for client_id in train_msg_buffer.keys():
                msg_list.append(train_msg_buffer[client_id])
                if self._cfg.fedtfp.gd_use:
                    train_data_size, gradient = train_msg_buffer[client_id]
                    if self.state >= self.last_gradients_length:
                        similarity = F.cosine_similarity(median_gradient,
                                                         torch.cat([para.flatten(0) for para in gradient.values()]),
                                                         dim=0).item()
                        if similarity < self.unreliable_gradients_threshold:
                            unreliable_clients[client_id] = 1
                    self.last_gradients_lists[client_id - 1].append(gradient)
            self._monitor.calc_model_metric(self.model.state_dict(), msg_list, rnd=self.state)
            agg_info = {'client_feedback': msg_list, 'recover_fun': None}
            result = self.aggregator.aggregate(agg_info)
            merged_param = merge_param_dict(self.model.state_dict().copy(), result)
            self.model.load_state_dict(merged_param, strict=False)

        if self._cfg.fedtfp.gd_use:
            return unreliable_clients

    def broadcast_model_para(self, msg_type='model_para', unreliable_clients=[]):
        if self._cfg.fedtfp.mg_use and msg_type == 'model_para':
            model_para = [model.state_dict() for model in self.models_fedtfp]
            for i in range(self.model_num_fedtfp):
                if self._cfg.fedtfp.gd_use:
                    self.comm_manager.send(Message(
                        msg_type=msg_type, sender=self.ID, receiver=self.clusters_fedtfp[i],
                        state=min(self.state, self.total_round_num), content=(unreliable_clients[i], model_para[i])))
                else:
                    self.comm_manager.send(Message(
                        msg_type=msg_type, sender=self.ID, receiver=self.clusters_fedtfp[i],
                        state=min(self.state, self.total_round_num), content=model_para[i]))
        else:
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')
            # We define the evaluation happens at the end of an epoch
            rnd = self.state - 1 if msg_type == 'evaluate' else self.state
            self.comm_manager.send(
                Message(msg_type=msg_type, sender=self.ID, receiver=receiver, state=min(rnd, self.total_round_num),
                        content=self.model.state_dict()))

    def check_buffer(self, cur_round, min_received_num, check_eval_result=False):
        """
        To check the message buffer

        Arguments:
            cur_round (int): The current round number
            min_received_num (int): The minimal number of the receiving \
                messages
            check_eval_result (bool): To check training results for \
                evaluation results

        Returns
            bool: Whether enough messages have been received or not
        """
        if check_eval_result:
            if 'eval' not in self.msg_buffer.keys() or len(self.msg_buffer['eval'].keys()) == 0:
                return False
            buffer = self.msg_buffer['eval']
            cur_round = max(buffer.keys())
            cur_buffer = buffer[cur_round]
            return len(cur_buffer) >= min_received_num
        else:
            if cur_round not in self.msg_buffer['train']:
                cur_buffer = dict()
            else:
                cur_buffer = self.msg_buffer['train'][cur_round]
            return len(cur_buffer) >= min_received_num

    def check_client_join_in(self):
        if len(self._cfg.federate.join_in_info) != 0:
            return len(self.join_in_info) == self.client_num
        else:
            return self.join_in_client_num == self.client_num

    def terminate(self, msg_type='finish'):
        self.is_finish = True
        self._monitor.finish_fl()
        self.comm_manager.send(
            Message(msg_type=msg_type, sender=self.ID, receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state, content=self.model.state_dict()))

    def callback_funcs_model_para(self, message: Message):
        if self.is_finish:
            return 'finish'
        round = message.state
        sender = message.sender
        content = message.content
        self.sampler.change_state(sender, 'idle')
        # update the currency timestamp according to the received message
        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content
        else:
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date message from round #{round}')
        move_on_flag = self.check_and_move_on()
        return move_on_flag

    def callback_funcs_for_join_in(self, message: Message):
        self.join_in_client_num += 1
        sender, address = message.sender, message.content
        self.comm_manager.add_neighbors(neighbor_id=sender, address=address)
        if self.check_client_join_in():
            self.broadcast_model_para(msg_type='model_para')
            logger.info('----------- Starting training (Round #{:d}) -------------'.format(self.state))

    def callback_funcs_for_metrics(self, message: Message):
        """
        The handling function for receiving the evaluation results, \
        which triggers ``check_and_move_on`` (perform aggregation when \
        enough feedback has been received).

        Arguments:
            message: The received message
        """
        rnd = message.state
        sender = message.sender
        content = message.content
        if rnd not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][rnd] = dict()
        self.msg_buffer['eval'][rnd][sender] = content
        return self.check_and_move_on(check_eval_result=True)

    def save_client_eval_results(self):
        """
        save the evaluation results of each client when the fl course terminated
        """
        rnd = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][rnd]
        with open(os.path.join(self._cfg.outdir, "eval_results.log"), "a") as outfile:
            for client_id, client_eval_results in eval_msg_buffer.items():
                formatted_res = self._monitor.format_eval_res(client_eval_results, rnd=self.state,
                                                              role='Client #{}'.format(client_id), return_raw=True)
                logger.info(formatted_res)
                outfile.write(str(formatted_res) + "\n")

    def merge_eval_results_from_all_clients(self):
        """
        Merge evaluation results from all clients, update best, \
        log the merged results and save them into eval_results.log

        Returns:
            the formatted merged results
        """
        round = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][round]
        eval_res_set = []
        for client_id in eval_msg_buffer:
            if eval_msg_buffer[client_id] is None:
                continue
            else:
                eval_res_set.append(eval_msg_buffer[client_id])
        formatted_logs_all_set = dict()
        if eval_res_set != []:
            metrics_all_clients = dict()
            for client_eval_results in eval_res_set:
                for key in client_eval_results.keys():
                    if key not in metrics_all_clients:
                        metrics_all_clients[key] = list()
                    metrics_all_clients[key].append(float(client_eval_results[key]))
            formatted_logs = self._monitor.format_eval_res(metrics_all_clients, rnd=round, role='Server #',
                                                           forms=self._cfg.eval.report)
            logger.info(formatted_logs)
            formatted_logs_all_set.update(formatted_logs)
            self._monitor.update_best_result(self.best_results, metrics_all_clients,
                                             results_type="client_best_individual")
            self._monitor.save_formatted_results(formatted_logs)
            for form in self._cfg.eval.report:
                if form != "raw":
                    metric_name = form
                    self._monitor.update_best_result(self.best_results, formatted_logs[f"Results_{metric_name}"],
                                                     results_type=f"client_summarized_{form}")
        return formatted_logs_all_set

    def save_best_results(self):
        """
        To Save the best evaluation results.
        """
        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(self._cfg.federate.save_to, self.state)
        formatted_best_res = self._monitor.format_eval_res(results=self.best_results, rnd="Final", role='Server #',
                                                           forms=["raw"], return_raw=True)
        logger.info(formatted_best_res)
        self._monitor.save_formatted_results(formatted_best_res)

    def check_and_save(self):
        if self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting merging results.')
            # last round
            self.save_best_results()
            if not self._cfg.federate.make_global_eval:
                self.save_client_eval_results()
            self.terminate(msg_type='finish')
        # Clean the clients evaluation msg buffer
        if not self._cfg.federate.make_global_eval:
            round = max(self.msg_buffer['eval'].keys())
            self.msg_buffer['eval'][round].clear()
        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            self.state += 1

    def _merge_and_format_eval_results(self):
        formatted_eval_res = self.merge_eval_results_from_all_clients()
        self.history_results = merge_dict_of_results(self.history_results, formatted_eval_res)
        if self.mode == 'standalone' and self._monitor.wandb_online_track and self._monitor.use_wandb:
            self._monitor.merge_system_metrics_simulation_mode(file_io=False, from_global_monitors=True)
        self.check_and_save()

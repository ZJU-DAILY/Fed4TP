import time
import pywt
from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.core.workers.base_client import BaseClient
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Client(BaseClient):
    def __init__(self, ID=-1, server_id=None, state=-1, config=None, data=None, model=None, device='cpu', strategy=None,
                 *args, **kwargs):
        super(Client, self).__init__(ID, state, config, model, strategy)
        self.data = data
        # Register message handlers
        self._register_default_handlers()
        # Parse the attack_id since we support both 'int' (for single attack)
        # and 'list' (for multiple attacks) for config.attack.attack_id
        parsed_attack_ids = list()
        if isinstance(config.attack.attacker_id, int):
            parsed_attack_ids.append(config.attack.attacker_id)
        elif isinstance(config.attack.attacker_id, list):
            parsed_attack_ids = config.attack.attacker_id
        else:
            raise TypeError(f"The expected types of config.attack.attack_id include 'int' and 'list', "
                            f"but we got {type(config.attack.attacker_id)}")
        # Attack only support the standalone model;
        # Check if is a attacker; a client is a attacker if the
        # config.attack.attack_method is provided
        self.is_attacker = ID in parsed_attack_ids and config.attack.attack_method != ''
        # Build Trainer
        # trainer might need configurations other than those of trainer node
        if self._cfg.fedtfp.tm_use:
            self.time_window = 0
            self.trainer = get_trainer(model=model, data=data[self.time_window], device=device, config=self._cfg,
                                       is_attacker=self.is_attacker, monitor=self._monitor)
        else:
            self.trainer = get_trainer(model=model, data=data, device=device, config=self._cfg,
                                       is_attacker=self.is_attacker, monitor=self._monitor)
        self.device = device

        # For client-side evaluation
        self.best_results = dict()
        self.history_results = dict()
        self.msg_buffer = {'train': dict(), 'eval': dict()}

        # Initialize communication manager
        self.server_id = server_id
        comm_queue = kwargs['shared_comm_queue']
        if self._cfg.federate.process_num <= 1:
            self.comm_manager = StandaloneCommManager(comm_queue=comm_queue, monitor=self._monitor)
        else:
            self.comm_manager = StandaloneDDPCommManager(comm_queue=comm_queue, monitor=self._monitor)
        self.local_address = None

    def join_in(self):
        self.comm_manager.send(Message(msg_type='join_in', sender=self.ID, receiver=[self.server_id],
                                       content=self.local_address))

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        content = message.content

        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)
        if self._cfg.fedtfp.gd_use and self.state > -1:
            unreliable_clients = content[0]
            content = content[1]
            if self._cfg.fedtfp.mg_use:
                if unreliable_clients[(self._ID - 1) % 3]:
                    coeffs_filtered = [pywt.threshold(c, 0.5 * max(c)) for c in pywt.wavedec(self.data, 'db4', level=3)]
                    self.data = pywt.waverec(coeffs_filtered, 'db4')
            else:
                if unreliable_clients[self._ID - 1]:
                    coeffs_filtered = [pywt.threshold(c, 0.5 * max(c)) for c in pywt.wavedec(self.data, 'db4', level=3)]
                    self.data = pywt.waverec(coeffs_filtered, 'db4')
        self.trainer.update(content, strict=self._cfg.federate.share_local_model)
        self.state = round
        sample_size, model_para_all, results = self.trainer.train()
        train_log_res = self._monitor.format_eval_res(results, rnd=self.state, role='Client #{}'.format(self.ID),
                                                      return_raw=True)
        logger.info(train_log_res)
        if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
            self._monitor.save_formatted_results(train_log_res, save_file_name="")

        shared_model_para = model_para_all
        self.comm_manager.send(Message(msg_type='model_para', sender=self.ID, receiver=[sender], state=self.state,
                                       content=(sample_size, shared_model_para)))

        if self._cfg.fedtfp.tm_use:
            if self.state == self._cfg.federate.total_round_num - 1 and \
                    self.time_window < self._cfg.fedtfp.time_window_num - 1:
                self.time_window += 1
                self.trainer.ctx.data = self.data[self.time_window]

    def callback_funcs_for_finish(self, message: Message):
        logger.info(f"================= client {self.ID} received finish message =================")
        if message.content is not None:
            self.trainer.update(message.content, strict=False)
        self._monitor.finish_fl()

    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender = message.sender
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content, strict=False)
        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(target_data_split_name=split)
            metrics.update(**eval_metrics)
        formatted_eval_res = self._monitor.format_eval_res(metrics, rnd=self.state, role='Client #{}'.format(self.ID),
                                                           forms=['raw'], return_raw=True)
        self._monitor.update_best_result(self.best_results, formatted_eval_res['Results_raw'],
                                         results_type=f"client #{self.ID}")
        self.history_results = merge_dict_of_results(self.history_results, formatted_eval_res['Results_raw'])
        self.comm_manager.send(Message(msg_type='metrics', sender=self.ID, receiver=[sender],
                                       state=self.state, content=metrics))

from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.feat_engr_builder import get_feat_engr_wrapper
from collections import deque
import abc
import logging
import heapq
import torch

logger = logging.getLogger(__name__)


class BaseRunner(object):
    def __init__(self, data, server_class=Server, client_class=Client, config=None, client_configs=None):
        self.data = data
        self.cfg = config
        self.serial_num_for_msg = 0
        self.mode = self.cfg.federate.mode.lower()
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu, specified_device=self.cfg.device)
        self.feat_engr_wrapper_client, self.feat_engr_wrapper_server = get_feat_engr_wrapper(config)
        self._set_up()

    @abc.abstractmethod
    def _set_up(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_server_args(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_client_args(self, client_id):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    def _setup_server(self):
        self.server_id = 0
        server_data, model, kw = self._get_server_args()
        server = Server(ID=self.server_id, config=self.cfg, data=server_data, model=model,
                        client_num=self.cfg.federate.client_num, total_round_num=self.cfg.federate.total_round_num,
                        device=self.gpu_manager.auto_choice(), **kw)
        logger.info('Server has been set up ... ')
        return self.feat_engr_wrapper_server(server)

    def _setup_client(self, client_id=-1):
        self.server_id = 0
        client_data, kw = self._get_client_args(client_id)
        if self.cfg.fedtfp.tm_use:
            client = Client(ID=client_id, server_id=self.server_id, config=self.cfg, data=client_data,
                            model=get_model(self.cfg.model, client_data[0], backend=self.cfg.backend),
                            device=self.gpu_manager.auto_choice(), is_unseen_client=False, **kw)
        else:
            client = Client(ID=client_id, server_id=self.server_id, config=self.cfg, data=client_data,
                            model=get_model(self.cfg.model, client_data, backend=self.cfg.backend),
                            device=self.gpu_manager.auto_choice(), is_unseen_client=False, **kw)
        logger.info(f'Client {client_id} has been set up ... ')
        return self.feat_engr_wrapper_client(client)


class StandaloneRunner(BaseRunner):
    def _set_up(self):
        self.shared_comm_queue = deque()
        torch.set_num_threads(1)
        self.server = self._setup_server()
        self.client = {client_id: self._setup_client(client_id=client_id)
                       for client_id in range(1, self.cfg.federate.client_num + 1)}

    def _get_server_args(self):
        server_data = None
        data_representative = self.data[1]
        if self.cfg.fedtfp.tm_use:
            data_representative = data_representative[0]
        model = get_model(self.cfg.model, data_representative, backend=self.cfg.backend)
        kw = {'shared_comm_queue': self.shared_comm_queue, 'resource_info': None, 'client_resource_info': None}
        return server_data, model, kw

    def _get_client_args(self, client_id=-1):
        client_data = self.data[client_id]
        kw = {'shared_comm_queue': self.shared_comm_queue, 'resource_info': None}
        return client_data, kw

    def run(self):
        for each_client in self.client:
            self.client[each_client].join_in()
        server_msg_cache = list()
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if msg.receiver == [self.server_id]:
                    msg.serial_num = self.serial_num_for_msg
                    self.serial_num_for_msg += 1
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                self._handle_msg(msg)
            else:
                break

    def _handle_msg(self, msg, rcv=-1):
        if rcv != -1:
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return
        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes()
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0:
                self.server.msg_handlers[msg.msg_type](msg)
                self.server._monitor.track_download_bytes(download_bytes)
            else:
                self.client[each_receiver].msg_handlers[msg.msg_type](msg)
                self.client[each_receiver]._monitor.track_download_bytes(
                    download_bytes)


class DistributedRunner(BaseRunner):
    def _set_up(self):
        self.server_address = {'host': self.cfg.distribute.server_host, 'port': self.cfg.distribute.server_port}
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server()
        elif self.cfg.distribute.role == 'client':
            self.client_address = {'host': self.cfg.distribute.client_host, 'port': self.cfg.distribute.client_port}
            self.client = self._setup_client(self.cfg.distribute.data_idx)

    def _get_server_args(self):
        server_data = self.data[1]
        model = get_model(self.cfg.model, server_data, backend=self.cfg.backend)
        kw = self.server_address
        kw.update({'resource_info': None})
        return server_data, model, kw

    def _get_client_args(self, client_id):
        client_data = self.data[client_id]
        kw = self.client_address
        kw['server_host'] = self.server_address['host']
        kw['server_port'] = self.server_address['port']
        kw['resource_info'] = None
        return client_data, kw

    def run(self):
        if self.cfg.distribute.role == 'server':
            self.server.run_distributed()
        elif self.cfg.distribute.role == 'client':
            self.client.join_in()
            self.client.run_distributed()

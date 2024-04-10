import os
import argparse
from modules import resnet, network, contrastive_loss
from utils import yaml_config_hook
from torch.utils import data
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import copy


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward(retain_graph=True)
        optimizer.step()
        print(
            f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


    def permutation(x):
        x = torch.Tensor(x)
        perm_index = torch.randperm(x.size(0))
        return x[perm_index]


    def affine_transformation(x):
        x = torch.Tensor(x)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float)
        grid = F.affine_grid(theta, x.size())
        augmented_tensor = F.grid_sample(x, grid)
        return augmented_tensor.squeeze()


    def perturbation(x, perturbation_strength=0.1):
        x = torch.Tensor(x)
        noise = torch.randn_like(x)
        return x + perturbation_strength * noise


    def mask(x, mask_ratio=0.2):
        x = torch.Tensor(x)
        mask_value = 0.0
        num_masked_elements = int(mask_ratio * x.numel())
        masked_indices = random.sample(range(x.numel()), num_masked_elements)
        x.view(-1)[masked_indices] = mask_value
        return x


    def smooth(x):
        x = torch.Tensor(x)
        window_size = 10
        smooth_factor = 0.5

        def moving_average(data, window_size, smooth_factor):
            smoothed_data = data.clone()
            for i in range(data.size(0)):
                for j in range(data.size(1)):
                    for k in range(1, window_size):
                        if i - k >= 0:
                            smoothed_data[i, j] = smooth_factor * smoothed_data[i - k, j] + (1 - smooth_factor) * \
                                                  smoothed_data[i, j]
            return smoothed_data

        return moving_average(x, window_size, smooth_factor)


    class Feature(nn.Module):
        def __init__(self, input_size):
            super(Feature, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(input_size, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            x = self.flatten(torch.unsqueeze(x, 0))
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x


    model1 = Feature(16992 * 51)
    model2 = Feature(51 * 51)
    model3 = Feature(10)

    num_nodes_per_client = 51
    adj_ms = open("data/PeMS04.csv", 'r')
    adj_ms.readline()
    lines = adj_ms.readlines()
    adjs = []
    for client_idx in range(6):
        adj_ms_client = torch.zeros((num_nodes_per_client, num_nodes_per_client))
        for line in lines:
            p = line.split(',')
            if client_idx * num_nodes_per_client <= int(p[0]) < (client_idx + 1) * num_nodes_per_client and \
                    client_idx * num_nodes_per_client <= int(p[1]) < (client_idx + 1) * num_nodes_per_client:
                adj_ms_client[int(p[0]) - client_idx * num_nodes_per_client][
                    int(p[1]) - client_idx * num_nodes_per_client] = 1
                adj_ms_client[int(p[1]) - client_idx * num_nodes_per_client][
                    int(p[0]) - client_idx * num_nodes_per_client] = 1
        adjs.append([torch.Tensor(adj_ms_client), permutation(copy.deepcopy(adj_ms_client)),
                     affine_transformation(copy.deepcopy(adj_ms_client)),
                     mask(copy.deepcopy(adj_ms_client), 0.4)])

    dataset = np.load("data/PeMS04.npy").astype('float32')[..., :306, 0]
    datasets = []
    for client_idx in range(6):
        d = dataset[:, client_idx * 51: (client_idx + 1) * 51]
        datasets.append(
            [torch.Tensor(d), perturbation(copy.deepcopy(d), 0.1), mask(copy.deepcopy(d), 0.1)])

    catering = ['bar', 'biergarten', 'cafe', 'fast_food',
                'food_court', 'ice_cream', 'pub', 'restaurant']
    education = ['college', 'dancing_school', 'driving_school', 'first_aid_school',
                 'kindergarten', 'language_school', 'library', 'surf_school',
                 'toy_library', 'research_institute', 'training', 'music_school',
                 'school', 'traffic_park', 'university']
    traffic = ['bicycle_parking', 'bicycle_repair_station', 'bicycle_rental',
               'bicycle_wash', 'boat_rental', 'boat_sharing', 'bus_station', 'car_rental',
               'car_sharing', 'car_wash', 'compressed_air', 'vehicle_inspection',
               'charging_station', 'driver_training', 'ferry_terminal', 'fuel', 'grit_bin',
               'motorcycle_parking', 'parking', 'parking_entrance', 'parking_space', 'taxi', 'weighbridge']
    finance = ['atm', 'payment_terminal', 'bank', 'bureau_de_change', 'money_transfer', 'payment_centre']
    medical_treatment = ['baby_hatch', 'clinic', 'dentist', 'doctors', 'hospital', 'nursing_home', 'pharmacy',
                         'social_facility', 'veterinary']
    entertainment = ['arts_centre', 'brothel', 'casino', 'cinema', 'community_centre', 'conference_centre',
                     'events_venue', 'exhibition_centre', 'fountain', 'gambling', 'love_hotel', 'music_venue',
                     'nightclub',
                     'planetarium', 'public_bookcase', 'social_centre', 'stripclub', 'studio', 'swingerclub', 'theatre']
    public = ['courthouse', 'fire_station', 'police', 'post_box', 'post_depot',
              'post_office', 'prison', 'ranger_station', 'townhall']
    facilities = ['bbq', 'bench', 'dog_toilet', 'dressing_room', 'drinking_water', 'give_box',
                  'mailroom', 'parcel_locker', 'shelter', 'shower', 'telephone', 'toilets', 'water_point',
                  'watering_place']
    waste_management = ['sanitary_dump_station', 'recycling', 'waste_basket', 'waste_disposal',
                        'waste_transfer_station']

    pois = []
    for i in range(6):
        all = [catering, education, traffic, finance, medical_treatment, entertainment, public, facilities,
               waste_management]

        all_num = [0 for _ in range(10)]

        file_path = 'data/POI/POI_frequency/poi' + str(i + 1) + '.xls'
        df = pd.read_excel(file_path)

        for index, row in df.iterrows():
            # print(f"Amenity: {row['amenity']}, Frequency: {row['FREQUENCY']}")
            flag = False
            for j in range(9):
                if row['amenity'] in all[j]:
                    all_num[j] += row['FREQUENCY']
                    flag = True
                    break
            if not flag:
                all_num[9] += row['FREQUENCY']
        poi = []
        for j in range(10):
            poi.append(all_num[j])
        pois.append([torch.Tensor(poi), perturbation(poi, 10)])

    features = []
    for client_idx in range(6):
        features.append([])
        for i in range(3):
            for j in range(4):
                for k in range(2):
                    features[client_idx].append(torch.cat(
                        (model1(datasets[client_idx][i]), model2(adjs[client_idx][j]), model3(pois[client_idx][k])),
                        dim=1))

    data1 = []
    data2 = []
    for client_idx in range(6):
        for f1 in features[client_idx]:
            for f2 in features[client_idx]:
                data1.append(f1)
                data2.append(f2)

    from torchvision.datasets.vision import VisionDataset
    from torch.utils import data


    class Traffic(VisionDataset):
        def __init__(self, data1, data2) -> None:
            super(Traffic, self).__init__('')
            self.data1 = data1
            self.data2 = data2
            self.data1 = torch.unsqueeze(torch.stack(self.data1, dim=0).resize(3456, 3, 64), -1)
            self.data2 = torch.unsqueeze(torch.stack(self.data2, dim=0).resize(3456, 3, 64), -1)

        def __getitem__(self, index: int):
            return [self.data1[index], self.data2[index]], len(self.data1)

        def __len__(self) -> int:
            return len(self.data1)


    traffic = Traffic(data1, data2)
    dataset = data.ConcatDataset([traffic])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    class_num = 6

    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(1, args.instance_temperature, loss_device).to(loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    for epoch in range(0, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")

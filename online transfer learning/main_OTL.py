import os
import numpy as np
import pickle
import torch
from gan_LSTM import LSTM
import scipy.stats as ss

# models and sensor
test_sensor = 50
model_dict = {607: 3, 612: 2}

# Parameters
input_dim = 14
hidden_dim = 500
output_dim = 1

# Number of models
N = len(model_dict.keys())
models = []
models_origin = []
# Load models
for car, epoch in model_dict.items():
    outputFile = 'result10/%d/%d_weights' % (car, test_sensor)
    weight_name = outputFile + '/model_%d.ckpt' % epoch
    hidden_name_h = outputFile + '/hidden_h_%d.pt' % epoch
    hidden_name_c = outputFile + '/hidden_c_%d.pt' % epoch
    net = LSTM(input_dim, hidden_dim, output_dim).cuda()
    hidden_state = (torch.load(hidden_name_h), torch.load(hidden_name_c))
    net.load_state_dict(torch.load(weight_name))
    models.append([net, hidden_state])

for car, epoch in model_dict.items():
    outputFile = 'result10/%d/%d_weights' % (car, test_sensor)
    weight_name = outputFile + '/model_%d.ckpt' % epoch
    hidden_name_h = outputFile + '/hidden_h_%d.pt' % epoch
    hidden_name_c = outputFile + '/hidden_c_%d.pt' % epoch
    net = LSTM(input_dim, hidden_dim, output_dim).cuda()
    hidden_state = (torch.load(hidden_name_h), torch.load(hidden_name_c))
    net.load_state_dict(torch.load(weight_name))
    models_origin.append([net, hidden_state])

# Sort in time order for valication files
def SortByDate(keys):
    ordered_list = []
    sort_dict = {}
    sort_list = []
    for key in keys:
        str_list = key.split('_')
        year = int(str_list[0])
        month = int(str_list[1])
        day = int(str_list[2])
        val = year * 1000 + month * 100 + day
        sort_dict[val] = key
        sort_list.append(val)
    sort_list.sort()
    for data in sort_list:
        ordered_list.append(sort_dict[data])
    return ordered_list

def SortByMin(keys):
    ordered_list = []
    sort_dict = {}
    sort_list = []
    for key in keys:
        str_list = key.split('_')
        minute = int(str_list[-1].split('.')[0])
        hour = int(str_list[-2])
        val = hour * 100 + minute
        sort_dict[val] = key
        sort_list.append(val)
    sort_list.sort()
    for data in sort_list:
        ordered_list.append(sort_dict[data])
    return ordered_list

def makeTrainingSet(training_path, training_pkl):
    files = os.listdir(training_path)
    all_data = []
    files = SortByDate(files)
    for file in files:
        pkl_files = os.listdir(os.path.join(training_path, file))
        pkl_files = SortByMin(pkl_files)
        for pkl_file in pkl_files:
            with open(os.path.join(training_path, file, pkl_file), 'rb') as f:
                one_sample = pickle.load(f)
                if len(one_sample) != 159:
                    continue
                if len(one_sample[0]) != 88:
                    continue
                one_sample = np.array(one_sample)
                if one_sample.shape != (159, 88):
                    continue
                all_data.append(one_sample)

    all_data = np.array(all_data)
    pickle.dump(all_data, open(training_pkl, 'ab'))
    return all_data

# Set target car
car = 614
training_path = '%d/trans' % car
training_pkl = 'trans%d.pkl' % car
validation_path = 'validation/%d' % car
training_bound = 'bound%d.pkl' % car

# Raw correct list
raw_correct_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 34, 41, 44, 60]

# Hyperparameters
v_Lr = 1e-5
u_Lr = 1e-5

correct_idx = []
problem_idx = []
for idx in raw_correct_idx:
    correct_idx.append(idx - 1)
correct_l = len(correct_idx)

for idx in [test_sensor]:
    problem_idx.append(idx - 1)
problem_l = len(problem_idx)

# Load input data as training set
print('Loading training data')
if os.path.exists(training_pkl):
    with open(training_pkl, 'rb') as f:
        training_data = pickle.load(f)
else:
    training_data = makeTrainingSet(training_path, training_pkl)

# Normalize input
if os.path.exists(training_bound):
    with open(training_bound, 'rb') as f:
        max_min_list = pickle.load(f)
else:
    max_min_list = [(np.max(training_data[:, :, a]), np.min(training_data[:, :, a])) for a in
                    range(training_data.shape[2])]
    with open(training_bound, 'wb') as f:
        pickle.dump(max_min_list, f)

for id in range(training_data.shape[2]):
    if max_min_list[id][0] == max_min_list[id][1]:
        training_data[:, :, id] *= 0.
        continue
    else:
        training_data[:, :, id] = ((training_data[:, :, id] - max_min_list[id][1]) / (
                    max_min_list[id][0] - max_min_list[id][1]) - 0.5) * 2
print('Training data loaded')

# Start training
Z = training_data[:, :, correct_idx]
X = training_data[:, :, problem_idx]

# Put np data to tensor
Z = torch.from_numpy(Z).to(dtype=torch.float32)
X = torch.from_numpy(X).to(dtype=torch.float32)

# Initialize new netwokr and optimizer
v_net = LSTM(input_dim, hidden_dim, output_dim).cuda()
v_optimizer = torch.optim.Adam(v_net.parameters(), lr=v_Lr)
u_optimizer_list = []
for i in range(N):
    u_optimizer_list.append(torch.optim.Adam(models[i][0].parameters(), lr=u_Lr))
L1Loss = torch.nn.MSELoss()
v_hidden_state = v_net.init_hidden()

# Initialize weights
u_v = [1 / (N + 1) for i in range(N + 1)]
beta = 0.9

u_v_compare = u_v

loss_raw = []
loss_new = []
u_v_list = []
u_v_compare_list = []
test_raw = []
test_new = []
test_new_compare = []

# Set test set
x_test = X[-1000:, :, :].cuda().transpose(2, 1)
z_test = Z[-1000:, :, :].cuda().transpose(0, 1)
target_epoch = 15
target_outputFile = 'result10/%d/%d_weights' % (car, test_sensor)
target_weight_name = target_outputFile + '/model_%d.ckpt' % target_epoch
target_hidden_name_h = target_outputFile + '/hidden_h_%d.pt' % target_epoch
target_hidden_name_c = target_outputFile + '/hidden_c_%d.pt' % target_epoch
target_net = LSTM(input_dim, hidden_dim, output_dim).cuda()
target_hidden_state = (torch.load(target_hidden_name_h), torch.load(target_hidden_name_c))
target_net.load_state_dict(torch.load(target_weight_name))

v_x_fake, target_hidden_state = target_net(z_test, target_hidden_state)
v_x_fake = v_x_fake.transpose(0, 1)
regression_loss = L1Loss(v_x_fake, x_test)
target_loss = regression_loss.cpu().detach().numpy()

# Start online training
# for d in range(training_data.shape[0]):
for d in range(500):
    # Use new data to train the model firstly
    x_batch = X[d:d + 1, :, :].cuda().transpose(2, 1)
    z_batch = Z[d:d + 1, :, :].cuda().transpose(0, 1)
    v_x_fake, v_hidden_state = v_net(z_batch, v_hidden_state)

    # test
    v_x_test, v_hidden_state = v_net(z_test, v_hidden_state)
    v_x_test = v_x_test.transpose(0, 1)
    test_loss = L1Loss(v_x_test, x_test)
    v_loss_test = test_loss.cpu().detach().numpy()

    v_x_fake = v_x_fake.transpose(0, 1)
    regression_loss = L1Loss(v_x_fake, x_batch)
    v_loss = regression_loss.cpu().detach().numpy()
    v_optimizer.zero_grad()
    regression_loss.backward(retain_graph=True)
    v_optimizer.step()

    # Make prediction and update
    u_loss_list = []
    u_loss_test_list = []
    u_loss_test_compare_list = []
    for i in range(N):
        u_net = models[i][0]
        u_hidden_state = models[i][1]
        u_optimizer = u_optimizer_list[i]

        # test
        u_x_fake_test, u_hidden_state = u_net(z_test, u_hidden_state)
        u_x_fake_test = u_x_fake_test.transpose(0, 1)
        regression_loss_test = L1Loss(u_x_fake_test, x_test)
        u_loss_test_list.append(regression_loss_test.cpu().detach().numpy().copy())

        u_x_fake, u_hidden_state = u_net(z_batch, u_hidden_state)
        u_x_fake = u_x_fake.transpose(0, 1)
        regression_loss = L1Loss(u_x_fake, x_batch)
        u_loss_list.append(regression_loss.cpu().detach().numpy().copy())
        u_optimizer.zero_grad()
        regression_loss.backward(retain_graph=True)
        u_optimizer.step()

        models[i][0] = u_net
        models[i][1] = u_hidden_state

        # Test with the original model
        u_net = models_origin[i][0]
        u_hidden_state = models_origin[i][1]

        # test
        u_x_fake_test, u_hidden_state = u_net(z_test, u_hidden_state)
        u_x_fake_test = u_x_fake_test.transpose(0, 1)
        regression_loss_test = L1Loss(u_x_fake_test, x_test)
        u_loss_test_compare_list.append(regression_loss_test.cpu().detach().numpy().copy())

    # Update
    sum_u_v = sum(u_v)
    u_v_new = [float(i) / sum_u_v for i in u_v]
    u_v = u_v_new
    u_v_list.append(u_v.copy())
    total_loss = 0
    for i in range(N):
        total_loss += u_v[i] * u_loss_list[i]
    total_loss += u_v[-1] * v_loss

    total_loss_test = 0
    for i in range(N):
        total_loss_test += u_v[i] * u_loss_test_list[i]
    total_loss_test += u_v[-1] * v_loss_test
    # Rank the models and updated u and v
    u_loss_test_list.append(v_loss.copy())
    rank_idx = ss.rankdata(u_loss_test_list)
    for i in range(N + 1):
        u_v[i] = u_v[i] * (beta ** rank_idx[i])
    loss_new.append(total_loss.copy())
    loss_raw.append(v_loss.copy())
    test_new.append(total_loss_test.copy())
    test_raw.append(v_loss_test.copy())

    # Update compare
    sum_u_v = sum(u_v_compare)
    u_v_new = [float(i) / sum_u_v for i in u_v_compare]
    u_v_compare = u_v_new
    u_v_compare_list.append(u_v_compare.copy())

    total_loss_test = 0
    for i in range(N):
        total_loss_test += u_v_compare[i] * u_loss_test_compare_list[i]
    total_loss_test += u_v_compare[-1] * v_loss_test
    # Rank the models and updated u and v
    u_loss_test_compare_list.append(v_loss.copy())
    rank_idx = ss.rankdata(u_loss_test_compare_list)
    for i in range(N + 1):
        u_v_compare[i] = u_v_compare[i] * (beta ** rank_idx[i])
    test_new_compare.append(total_loss_test.copy())
    a = 1
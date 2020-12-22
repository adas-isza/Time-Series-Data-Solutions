import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         # self.batch_size = batch_size
#         self.c1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=1,
#                 out_channels=32,
#                 kernel_size=5,
#                 stride=2,
#                 padding=2,
#                 bias=False
#             ),
#             nn.LeakyReLU(0.01),
#             )
#         self.c2 = nn.Sequential(
#             nn.Conv1d(32, 64, 5, 2, 2),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm1d(64),
#             )
#         # self.c3 = nn.Sequential(
#         #     nn.Conv1d(32, 64, 5, 2, 2),
#         #     nn.LeakyReLU(0.01),
#         #     nn.BatchNorm1d(64),
#         #     )
#
#         self.c3 = nn.Sequential(
#             nn.Conv1d(64, 128, 3, 2, 1),
#             nn.LeakyReLU(0.01),
#             nn.BatchNorm1d(128),
#             )
#         self.f1 = nn.Sequential(
#             nn.Linear(128*20, 256),
#             nn.LeakyReLU(0.01),
#         )
#         self.f2 = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(),
#         )
#         self.f3 = nn.Sequential(
#             nn.Linear(64, 1),
#         )
#         self.classify = nn.Sigmoid()
#
#     def forward(self, input):
#         x = self.c1(input)
#         x = self.c2(x)
#         x = self.c3(x)
#         # x = self.c4(x)
#         x = self.f1(x.view(x.size(0), -1))
#         x = self.f2(x)
#         x = self.f3(x)
#         output = self.classify(x)
#         return output
#
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.l1 = nn.Sequential(
#             nn.Linear(35, 256),
#             # nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.01),
#             )
#         self.l2 = nn.Sequential(
#             nn.Linear(256, 256),
#             # nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.01),
#             )
#         self.l3 = nn.Sequential(
#             nn.Linear(256, 64),
#             # nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.01),
#             )
#         self.l4 = nn.Sequential(
#             nn.Linear(64, 1),
#             nn.Tanh()
#             )
#
#     def forward(self, input):
#         x = self.l1(input)
#         x = self.l2(x)
#         x = self.l3(x)
#         output = self.l4(x)
#         return output
#
#
# class Generator_LSTM(nn.Module):
#     def __init__(self, hidden_layer):
#         super(Generator_LSTM, self).__init__()
#         self.hidden_layer = hidden_layer
#         self.lstm = nn.LSTM(input_size=35, hidden_size=self.hidden_layer, num_layers=1)
#         self.tanh = nn.Tanh()
#         self.l = nn.Linear(hidden_layer, 1)
#
#     def hidden_init(self, Batch_size, hidden_layer):
#         return (torch.zeros(1, Batch_size, hidden_layer).cuda(), torch.zeros(1, Batch_size, hidden_layer).cuda())
#
#     def forward(self, z, h_state):
#         r_out, h_state = self.lstm(z, h_state)
#         r_out = self.l(r_out)
#         r_out = self.tanh(r_out)
#         return r_out, h_state
#
# class Discriminator_LSTM(nn.Module):
#     def __init__(self, hidden_layer):
#         super(Discriminator_LSTM, self).__init__()
#         self.hidden_layer = hidden_layer
#         self.lstm = nn.LSTM(input_size=40, hidden_size=self.hidden_layer, num_layers=1)
#         self.l = nn.Sequential(
#             nn.Linear(hidden_layer, 40),
#             nn.Tanh(40),
#         )
#
#     def hidden_init(self, Batch_size, hidden_layer):
#         return (torch.zeros(1, Batch_size, hidden_layer).cuda(), torch.zeros(1, Batch_size, hidden_layer).cuda())
#
#     def forward(self, z, h_state):
#         r_out, h_state = self.lstm(z, h_state)
#         r_out = self.l(r_out)
#         return r_out, h_state
#
# class Discriminator_old(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.c0 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=40,
#                 out_channels=64,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2,
#                 bias=False
#             ),
#             nn.LeakyReLU(0.01),
#             )
#         self.c1 = nn.Sequential(
#             nn.Conv1d(64, 128, 5, 2, 2),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.01),
#             )
#         self.uc1 = nn.Sequential(
#             nn.Conv1d(128, 128, 5, 1, 2),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.01),
#             nn.ConvTranspose1d(128, 64, 5, 2, 2, 1)
#         )
#         self.uc2 = nn.Sequential(
#             nn.Conv1d(64, 64, 5, 1, 2),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.01),
#             nn.ConvTranspose1d(64, 40, 5, 2, 2, 1)
#         )
#         self.uc3 = nn.Sequential(
#             nn.Conv1d(40, 40, 5, 1, 2),
#         )
#         self.c2 = nn.Sequential(
#             nn.Conv1d(256, 512, 3, 2, 1),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.01),
#         )
#         # self.c3 = nn.Sequential(
#         #     nn.Conv1d(512, 1024, 3, 2, 1),
#         #     nn.BatchNorm1d(1024),
#         #     nn.LeakyReLU(0.01),
#         # )
#         self.center = nn.Sequential(
#             nn.Conv1d(512, 512, 3, 1, 1),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.01),
#             nn.ConvTranspose1d(512, 512, 5, 2, 2, 1)
#         )
#         # self.u1 = nn.Sequential(
#         #     nn.Conv1d(1536, 512, 3, 1, 1),
#         #     nn.BatchNorm1d(512),
#         #     nn.LeakyReLU(0.01),
#         #     nn.ConvTranspose1d(512, 512, 5, 2, 2, 1)
#         #
#         # )
#         self.u2 = nn.Sequential(
#             nn.Conv1d(768, 256, 3, 1, 1),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.01),
#             nn.ConvTranspose1d(256, 256, 5, 2, 2, 1)
#         )
#         self.classify = nn.Sequential(
#             nn.Conv1d(384, 40, 3, 1, 1),
#             nn.Sigmoid()
#
#         )
#
#     def forward(self, input):
#         raw = self.c0(input)
#         down1 = self.c1(raw)
#         up1 = self.uc1(down1)
#         up2 = self.uc2(up1)
#         output = self.uc3(up2)
#         return output
#
# class LSTM_all(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1, num_layers=2):
#         super(LSTM_all, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#         self.output_dim = output_dim
#         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
#         # Define output layer
#         self.linear = nn.Sequential(
#             nn.Linear(self.hidden_dim, 500),
#             nn.LeakyReLU(0.01),
#             nn.Linear(500, 500),
#             nn.LeakyReLU(0.01),
#             nn.Linear(500, self.output_dim),
#             nn.Tanh()
#         )
#
#     def init_hidden(self):
#         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
#
#     def forward(self, input, hidden_state):
#         # shape of input  (seq_length, batch, input_dim)
#         # shape of output (seq_length, batch, hidden_dim)
#         lstm_out, hidden_state = self.lstm(input)
#         self.hidden = hidden_state
#         y_pred = self.linear(lstm_out)
#         return y_pred, hidden_state


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # Define output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())

    def forward(self, input, hidden_state):
        # shape of input  (seq_length, batch, input_dim)
        # shape of output (seq_length, batch, hidden_dim)
        lstm_out, hidden_state = self.lstm(input)
        self.hidden = hidden_state
        y_pred = self.linear(lstm_out)
        y_pred = self.tanh(y_pred)
        return y_pred, hidden_state

# class LSTM_D(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1, num_layers=2):
#         super(LSTM_D, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#         self.output_dim = output_dim
#         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
#         # Define output layer
#         self.linear = nn.Linear(self.hidden_dim, self.output_dim)
#         self.classify = nn.Sigmoid()
#
#     def init_hidden(self):
#         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
#
#     def forward(self, input, hidden_state):
#         # shape of input  (seq_length, batch, input_dim)
#         # shape of output (seq_length, batch, hidden_dim)
#         lstm_out, hidden_state = self.lstm(input)
#         self.hidden = hidden_state
#         y_pred = self.linear(lstm_out)
#         y_pred = self.classify(y_pred)
#         y_pred = y_pred.mean()
#         return y_pred, hidden_state
#
#
# # class LSTM_D(nn.Module):
# #     def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1, num_layers=2):
# #         super(LSTM_D, self).__init__()
# #         self.input_dim = input_dim
# #         self.hidden_dim = hidden_dim
# #         self.batch_size = batch_size
# #         self.num_layers = num_layers
# #         self.output_dim = output_dim
# #         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
# #         # Define output layer
# #         self.linear = nn.Linear(self.hidden_dim, self.output_dim)
# #         self.classify = nn.Sigmoid()
# #
# #     def init_hidden(self):
# #         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
# #
# #     def forward(self, input, hidden_state):
# #         # shape of input  (seq_length, batch, input_dim)
# #         # shape of output (seq_length, batch, hidden_dim)
# #         lstm_out, hidden_state = self.lstm(input)
# #         self.hidden = hidden_state
# #         y_pred = self.linear(lstm_out)
# #         y_pred = self.classify(y_pred)
# #         y_pred = y_pred.mean()
# #         return y_pred, hidden_state



class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        # self.batch_size = batch_size
        self.c1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.LeakyReLU(0.01),
            )
        self.c2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),
            )
        self.c3 = nn.Sequential(
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(128),
            )
        self.f1 = nn.Sequential(
            nn.Linear(128*20, 516),
            nn.LeakyReLU(0.01),
        )
        self.f2 = nn.Sequential(
            nn.Linear(516, 64),
            nn.LeakyReLU(0.01),
        )
        self.f3 = nn.Sequential(
            nn.Linear(64, 1),
            nn.LeakyReLU(0.01)
        )
        self.classify = nn.Sigmoid()

    def forward(self, input):
        x = self.c1(input)
        x = self.c2(x)
        x = self.c3(x)
        # x = self.c4(x)
        x = self.f1(x.view(x.size(0), -1))
        x = self.f2(x)
        x = self.f3(x)
        output = self.classify(x)
        return output


class D_l(nn.Module):
    def __init__(self):
        super(D_l, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(159, 159*64),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            )
        # self.l2 = nn.Sequential(
        #     nn.Linear(159*256, 159*256),
        #     # nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.01),
        #     )
        self.l3 = nn.Sequential(
            nn.Linear(159*64, 159*64),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            )
        self.l4 = nn.Sequential(
            nn.Linear(159*64, 159*1),
            nn.Sigmoid()
            )

    def forward(self, input):
        x = self.l1(input)
        # x = self.l2(x)
        x = self.l3(x)
        output = self.l4(x)
        return output.mean()

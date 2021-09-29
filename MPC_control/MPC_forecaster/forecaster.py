import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class SimpleForecaster(nn.Module):
    def __init__(self, model_params, use_bias = True):
        super(SimpleForecaster, self).__init__()

        input_dim = model_params['H'] * model_params['s_dim']
        output_dim = model_params['W'] * model_params['s_dim'] 
        z_dim = model_params['z_dim']

        self.A_encoder = nn.Linear(input_dim, z_dim, bias = use_bias)
        self.B_decoder = nn.Linear(z_dim, output_dim, bias = use_bias)

        self.name = "SimpleForecaster"

    def forward(self, s):
        # s: (batch_size, W * forecast_dim)
        # z: (batch_size, z_dim)
        # s_hat: (batch_size, H * forecast_dim)

        z = self.A_encoder(s)
        s_hat = self.B_decoder(z)
        return s_hat


class DNNForecaster(nn.Module):
    def __init__(self, model_params, use_bias = True):
        super(DNNForecaster, self).__init__()

        input_dim = model_params['H'] * model_params['s_dim']
        output_dim = model_params['W'] * model_params['s_dim']
        hidden_dim = model_params['hidden_dim']
        z_dim = model_params['z_dim']

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias = use_bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias = use_bias)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias = use_bias)
        self.encoder = nn.Linear(output_dim, z_dim, bias = use_bias)
        self.decoder = nn.Linear(z_dim, output_dim, bias = use_bias)

        self.name = "DNNForecaster"

    def forward(self, s):
        # s: (batch_size, W * forecast_dim)
        # s_hat: (batch_size, H * forecast_dim)

        z = F.relu(self.fc1(s))
        z = F.relu(self.fc2(z))
        s_hat = F.relu(self.fc3(z))
        
        z = self.encoder(s_hat)
        s_hat = self.decoder(z)

        return s_hat


class LSTMForecaster(nn.Module):
    def __init__(self, model_params, device, use_bias = True):
        super(LSTMForecaster, self).__init__()

        self.W = model_params['W']
        self.H = model_params['H']

        self.s_dim = model_params['s_dim']
        self.hidden_dim = model_params['hidden_dim']

        self.z_dim = model_params['z_dim']

        self.lstm = nn.LSTM(self.s_dim, self.hidden_dim, bias = use_bias)
        self.linear = nn.Linear(self.hidden_dim, self.s_dim, bias = use_bias)

        self.encoder = nn.Linear(self.s_dim * self.H, self.z_dim, bias = use_bias)
        self.decoder = nn.Linear(self.z_dim, self.s_dim * self.H, bias = use_bias)

        self.device = device

        self.name = "LSTMForecaster"

    def forward(self, _batched_input):
        # batched_input: (batch_size, W * input_dim)
        # batched_output: (batch_size, H * output_dim)

        batch_size = _batched_input.shape[0]
        assert(_batched_input.shape[1] == self.W * self.s_dim)
        
        batched_input = _batched_input.reshape(batch_size, self.W, self.s_dim)

        hidden = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

        # LSTM encoder
        for i in range(self.W):
            lstm_input = batched_input[:, i:i+1, :].transpose(0, 1)
            lstm_output, hidden = self.lstm(lstm_input, hidden)
            lstm_forecast = self.linear(lstm_output)

        # LSTM decoder
        # lstm_forecasts = lstm_forecast
        lstm_forecasts = torch.cat((lstm_input, lstm_forecast), 0) 
        for i in range(self.H-2):
            lstm_input = lstm_forecast
            lstm_output, hidden = self.lstm(lstm_input, hidden)
            lstm_forecast = self.linear(lstm_output)
            lstm_forecasts = torch.cat((lstm_forecasts, lstm_forecast), 0)

        lstm_forecasts = lstm_forecasts.transpose(0, 1)

        assert(lstm_forecasts.shape[0] == batch_size)
        assert(lstm_forecasts.shape[1] == self.H)
        assert(lstm_forecasts.shape[2] == self.s_dim)

        lstm_forecasts = lstm_forecasts.reshape(batch_size, self.H * self.s_dim)

        s_hat = self.decoder(self.encoder(lstm_forecasts))

        return s_hat
from inference import Separator
from utils import random_file

import os
import librosa
import numpy as np
import soundfile as sf
from pedalboard.io import AudioFile
import torch
from tqdm import tqdm
from lib import dataset
from lib import nets
from lib import spec_utils
from lib import utils
# web service variation
from flask import Flask, request, send_file
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS

# any name will work for internal identification
app = Flask(__name__)
CORS(app)
api = Api(app)

class remover(Resource):
    # methods go here
    def post(self):
        args_gpu = -1
        args_pretrained_model = 'models/baseline.pth'
        args_sr = None
        args_n_fft = 2048
        args_hop_length = 1024
        args_batchsize = 4
        args_cropsize = 256
        args_tta = False
        args_postprocess = False
        args_output_dir = "download"

        temp = request.files['input']
        filename = temp.filename # random_file(prefix="temp",extension="mp3")
        filepath = os.path.join(args_output_dir, filename)
        temp.save(filepath)
        
        print('loading model...', end=' ')
        device = torch.device('cpu')
        model = nets.CascadedNet(args_n_fft, 32, 128)
        model.load_state_dict(torch.load(args_pretrained_model, map_location=device))
        if args_gpu >= 0:
            if torch.cuda.is_available():
                device = torch.device('cuda:{}'.format(args_gpu))
                model.to(device)
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device('mps')
                model.to(device)
        print('done')

        print('loading wave source...', end=' ')
        X, sr = librosa.load(
            filepath, args_sr, False, dtype=np.float32, res_type='kaiser_fast')
        # basename = os.path.splitext(os.path.basename(args.input))[0]
        print('done')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        print('stft of wave source...', end=' ')
        X_spec = spec_utils.wave_to_spectrogram(X, args_hop_length, args_n_fft)
        print('done')

        sp = Separator(model, device, args_batchsize, args_cropsize, args_postprocess)

        if args_tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        # print('validating output directory...', end=' ')
        # output_dir = args.output_dir
        # if output_dir != "":  # modifies output_dir if theres an arg specified
        #     output_dir = output_dir.rstrip('/') + '/'
        # #    os.makedirs(output_dir, exist_ok=True)
        # print('done')

        # print('inverse stft of instruments...', end=' ')
        # wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
        # print('done')
        # sf.write('{}{}_Instruments.wav'.format(output_dir, basename), wave.T, sr)

        print('inverse stft of vocals...', end=' ')
        wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args_hop_length)
        print('done')
        filename = 'vocal_' + filename
        filepath = os.path.join(args_output_dir, filename)
        with AudioFile(filepath, "w", samplerate=sr, num_channels=2, quality=128,  # kilobits per second
        ) as f:
            f.write(wave)
        # sf.write(filepath, wave.T, sr)

        return send_file(filepath)
        # temp = open(filepath, "rb")
        # result = temp.read()
        # temp.close
        # return {'vocal': result}, 200  # return data with 200 OK 
    pass

api.add_resource(remover, '/remover') 
if __name__ == '__main__':
    app.run(port=4000,debug=True)  # run our Flask app
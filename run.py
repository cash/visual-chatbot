#
# python run.py image_path
# It creates an equal number of white and black games.
#

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'visdial.settings')
import django
django.setup()
from django.conf import settings

import argparse
import PyTorch
import PyTorchHelpers
from chat import constants


class CaptionMaker(object):
    def __init__(self):
        model_class = PyTorchHelpers.load_lua_class(
            constants.CAPTIONING_LUA_PATH, 'CaptioningTorchModel')
        self.model = model_class(
            constants.CAPTIONING_CONFIG['model_path'],
            constants.CAPTIONING_CONFIG['backend'],
            constants.CAPTIONING_CONFIG['input_sz'],
            constants.CAPTIONING_CONFIG['layer'],
            constants.CAPTIONING_CONFIG['seed'],
            constants.CAPTIONING_GPUID
        )
        self.input_sz = constants.CAPTIONING_CONFIG['input_sz']

    def caption(self, image_path):
        result = self.model.predict(image_path, self.input_sz, self.input_sz)
        return result['pred_caption']

class ChatBot(object):
    def __init__(self):
        model_class = PyTorchHelpers.load_lua_class(
            constants.VISDIAL_LUA_PATH, 'VisDialTorchModel')
        self.model = model_class(
            constants.VISDIAL_CONFIG['input_json'],
            constants.VISDIAL_CONFIG['load_path'],
            constants.VISDIAL_CONFIG['beamSize'],
            constants.VISDIAL_CONFIG['beamLen'],
            constants.VISDIAL_CONFIG['sampleWords'],
            constants.VISDIAL_CONFIG['temperature'],
            constants.VISDIAL_CONFIG['gpuid'],
            constants.VISDIAL_CONFIG['backend'],
            constants.VISDIAL_CONFIG['proto_file'],
            constants.VISDIAL_CONFIG['model_file'],
            constants.VISDIAL_CONFIG['maxThreads'],
            constants.VISDIAL_CONFIG['encoder'],
            constants.VISDIAL_CONFIG['decoder']
        )
        # caption for /opt/coco/train2014/COCO_train2014_000000581921.jpg
        self.history = ['a snowboarder is doing a trick on a mountain']

    def answer(self, question):
        result = self.model.predict(args.image_path, self.history, question)
        question = str(result['question'])
        answer = str(result['answer'])
        answer = answer.replace("<START>", "").replace("<END>", "")
        self.history.append(" {} ?   {} ".format(question, answer))
        return answer


parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="path to an image")
args = parser.parse_args()

#captioner = CaptionMaker()
chat_bot = ChatBot()

print("")
print("Ask questions. 'c' to get a caption. 'q' to quit")
while True:
    question = raw_input("> ")
    question = question.replace("?", "").lower()
    if 'q' == question:
        break
    elif 'c' == question:
        #print(captioner.caption(args.image_path))
        pass
    else:
        print(chat_bot.answer(question))
    print("")

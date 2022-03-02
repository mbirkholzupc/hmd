from __future__ import print_function
import os
import sys
import datetime
import configparser
from BTFM import proc_btfm
from BTFM import proc_btfm_parallel

from hmr_predictor import hmr_predictor

sys.path.append("../src/")
from utility import take_notes

# parse configures
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
tgt_path = conf.get('DATA', 'tgt_path')
lsp_path = conf.get('DATA', 'lsp_path')
lspet_path = conf.get('DATA', 'lspet_path')
upi_path = conf.get('DATA', 'upi_path')
coco_api_path = conf.get('DATA', 'coco_api_path')
coco_list_path = conf.get('DATA', 'coco_list_path')
btfm_path = conf.get('DATA', 'btfm_path')
btfm_base = conf.get('DATA', 'btfm_base')

c_time = datetime.datetime.now()
time_string = "%s-%02d:%02d:%02d" % (c_time.date(), c_time.hour, c_time.minute, c_time.second)
take_notes("start at %s\n" % time_string, "./data_log.txt", create_file = True)

VISUALIZE_JOINTS=False
#VISUALIZE_JOINTS=True
#PREVIEW_ONLY=True
PREVIEW_ONLY=False

p_train = 0
p_test = 0

# build all dirs if not exist
for i in [tgt_path + "train/", tgt_path + "train/img/", 
          tgt_path + "train/sil/", tgt_path + "train/para/", 
          tgt_path + "test/", tgt_path + "test/img/", 
          tgt_path + "test/sil/", tgt_path + "test/para/"]:
    if not os.path.exists(i):
        os.makedirs(i)
if VISUALIZE_JOINTS:
    for i in [tgt_path + "train/vis", tgt_path + "test/vis/"]:
        if not os.path.exists(i):
            os.makedirs(i)

# Create the HMR object
#hmr_pred = hmr_predictor()
hmr_pred = None

#p_train, p_test = proc_btfm(tgt_path + "train/", tgt_path + "test/",
#                                     p_train, p_test,
#                                     btfm_path, btfm_base, hmr_pred, visualize_joints=VISUALIZE_JOINTS,
#                                     preview=PREVIEW_ONLY)
p_train, p_test = proc_btfm_parallel(tgt_path + "train/", tgt_path + "test/",
                                     p_train, p_test,
                                     btfm_path, btfm_base, hmr_pred, visualize_joints=VISUALIZE_JOINTS,
                                     preview=PREVIEW_ONLY)

print('---- After BTFM ----')
print('p_train: %d' % p_train)
print('p_test: %d' % p_test)

print("All done")

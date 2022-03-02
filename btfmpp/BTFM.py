from __future__ import print_function 
from __future__ import division 
import sys
sys.path.append("../src/")
import numpy as np
import PIL.Image
import pickle
import json
from tqdm import trange
from tqdm import tqdm
from time import sleep
import renderer as rd
from scipy.io import loadmat
from mesh_edit import fast_deform_dja
from mesh_edit import fast_deform_dsa
from hmr_predictor import proc_sil, predict_pponly, hmr_predictor
from utility import make_trimesh
from utility import get_joint_move
from utility import get_achr_move
from utility import get_anchor_posi
from data_filter import lsp_filter
from utility import take_notes
from btfm_utils import get_joints_lspformat, plot_lsp_joints
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue, Value
from Queue import Empty, Full

NOTES_FILE='./btfm_data_log.txt'

SKIP=0
MIN_JOINTS_NEEDED=13
# COCO, 3DPW missing annotations on head, neck, so reduce number needed
MIN_JOINTS_NEEDED_COCO=11
MIN_JOINTS_NEEDED_3DPW=11
MIN_JOINTS_NEEDED_MI3=13

NUM_HMRPP_WORKERS=4
NUM_HMRPOST_WORKERS=16

def proc_btfm(train_dir, test_dir, train_id, test_id, btfm_json, base_dir, hmr_pred, visualize_joints=False, preview=False):
    
    faces = np.load("../predef/smpl_faces.npy")
    face_num = len(faces)
    
    renderer = rd.SMPLRenderer(face_path = 
                               "../predef/smpl_faces.npy")
    
    with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
        mesh_joint = pickle.load(fp)

    # Load dataset
    with open(btfm_json, 'rb') as infile:
        btfm_dataset=json.load(infile)
    
    num_btfm_samples=len(btfm_dataset)
    count_all = 0.
    count_work = 0.
    # Create a counter to keep track of stats per set
    counter={}
    counter['LSP']=0
    counter['LSPET']=0
    counter['MPII']=0
    counter['COCO']=0
    counter['3DPW']=0
    counter['MI3']=0
    counter['SSP']=0

    if preview:
        # make train set
        tr = trange(num_btfm_samples-SKIP, desc='Bar desc', leave=True)
        for i in tr:
            tr.set_description("BTFM - train part")
            tr.refresh() # to show immediately the update
            sleep(0.0001)  # Used to be 0.01
            
            count_all += 1

            cur_sample=btfm_dataset[i+SKIP]
            joints=get_joints_lspformat(cur_sample)

            # read sil
            try:
                # Check if silhouette exists (open file, but don't read, for speed)
                sil_img_file = PIL.Image.open(base_dir + cur_sample['silhouette'])
            except KeyError as e:
                continue

            # Make sure there are enough valid joints
            img = PIL.Image.open(base_dir + cur_sample['path'])
            width, height = img.size
            img.close()
            result = check_joints(joints, cur_sample['set'], (width, height))
            if result is False:
                continue


            # TODO: judge using filter. Will need to do something here or judge during BTFM dataset creation
            #result = lsp_filter(lsp_joints[:,:,i+SKIP], src_gt_sil)
            #if result is False:
            #    take_notes("LSP %04d BAN -1\n" % (i+SKIP+1), "./data_log.txt")
            #    continue
            take_notes("LSP %04d TRAIN %08d\n" % (i+SKIP+1, train_id), "./data_log.txt")
            train_id += 1
            count_work += 1
            counter[cur_sample['set']] = counter[cur_sample['set']]+1

    else:
        # make train set
        tr = trange(num_btfm_samples-SKIP, desc='Bar desc', leave=True)
        for i in tr:
            tr.set_description("BTFM - train part")
            tr.refresh() # to show immediately the update
            sleep(0.0001)  # Used to be 0.01
            
            count_all += 1

            cur_sample=btfm_dataset[i+SKIP]
            joints=get_joints_lspformat(cur_sample)

            # read sil
            try:
                src_gt_sil = np.array(PIL.Image.open(
                         base_dir + cur_sample['silhouette']))
            except KeyError as e:
                continue

            if cur_sample['set'] in ['LSPET', 'MPII', 'MI3']:
                # LSPET, MPII encode silhouette as 3-channel but only need 1
                # MI3 has foreground/background info in Red channel (channel 0)
                src_gt_sil = src_gt_sil[:,:,0]

            # Make sure there are enough valid joints
            img = PIL.Image.open(base_dir + cur_sample['path'])
            width, height = img.size
            img.close()
            result = check_joints(joints, cur_sample['set'], (width, height))
            if result is False:
                continue
            
            # TODO: judge using filter. Will need to do something here or judge during BTFM dataset creation
            #result = lsp_filter(lsp_joints[:,:,i+SKIP], src_gt_sil)
            #if result is False:
            #    take_notes("LSP %04d BAN -1\n" % (i+SKIP+1), "./data_log.txt")
            #    continue
            
            # read ori img
            ori_img = np.array(PIL.Image.open(
                      base_dir + cur_sample['path']))

            # hmr predict
            # TODO: MI3 will need a different bbox/silhouette
            if cur_sample['set'] == 'LSP':
                verts, cam, proc_para, std_img = hmr_pred.predict(ori_img)
            elif cur_sample['set'] == 'MI3':
                bbox=np.array(cur_sample['bbox'])
                verts, cam, proc_para, std_img = hmr_pred.predict(ori_img, 
                                                                  use_j_bbox=True, 
                                                                  j_bbox=bbox)
            else:
                verts, cam, proc_para, std_img = hmr_pred.predict(ori_img, 
                                                                  True, 
                                                                  src_gt_sil)
            
            # unnormalize std_img
            src_img = ((std_img+1).astype(np.float)/2.0*255).astype(np.uint8)
            
            # save img (used to be 1-based, shouldn't matter)
            img_file = "img/BTFM_%08d.png" % (i+SKIP)
            PIL.Image.fromarray(src_img).save(train_dir + img_file)
            
            # process sil
            #print('src_gt_sil: ' + str(src_gt_sil.shape) + ' ' + cur_sample['set'])
            gt_sil = proc_sil(src_gt_sil, proc_para)

            # get proj sil
            proj_sil = renderer.silhouette(verts = verts,
                                           cam = cam,
                                           img_size = src_img.shape,
                                           norm = False)

            # make TriMesh
            mesh = make_trimesh(verts, faces, compute_vn = True)
            vert_norms = mesh.vertex_normals()

            # get joint move
            unseen_mode = True if (cur_sample['set'] in ['LSPET','COCO','3DPW']) else False
            new_jv, _, joint_move, joint_posi = get_joint_move(verts, 
                                                   joints, 
                                                   proc_para,
                                                   mesh_joint,
                                                   unseen_mode=unseen_mode)
            joint_move = joint_move.flatten()

            # joint deform
            fd_ja = fast_deform_dja(weight = 10.0)
            ja_verts = fd_ja.deform(np.asarray(verts), new_jv)

            # get achr move
            proj_sil_ja = renderer.silhouette(verts = ja_verts,
                                           norm = False)

            _, achr_verts, achr_move = get_achr_move(gt_sil, 
                                                     ja_verts, 
                                                     vert_norms,
                                                     proj_sil_ja)
            achr_posi = get_anchor_posi(achr_verts)
            
            # save sil
            compose_sil = np.stack((gt_sil, proj_sil, proj_sil_ja))
            compose_sil = np.moveaxis(compose_sil, 0, 2)
            compose_sil = PIL.Image.fromarray(compose_sil.astype(np.uint8))
            compose_sil.save(train_dir + "sil/%08d.png" % train_id)

            # save para
            proc_para['end_pt'] = proc_para['end_pt'].tolist()
            proc_para['start_pt'] = proc_para['start_pt'].tolist()
            para = {"verts": verts.tolist(),
                    "vert_norms": vert_norms.tolist(),
                    "proc_para": proc_para,
                    "joint_move": joint_move.tolist(),
                    "joint_posi": joint_posi.tolist(),
                    "achr_move": achr_move.tolist(),
                    "achr_posi": achr_posi.tolist(),
                    "img_file": img_file,
                   }
            with open(train_dir + "para/%08d.json" % train_id, 'wb') as fp:
                json.dump(para, fp)

            if visualize_joints:
                #vis_filename=train_dir + "vis/BTFM_%08d.png" % train_id
                vis_filename=train_dir + "vis/BTFM_%08d.png" % i
                img_w_joints=plot_lsp_joints(ori_img, joints)
                #img255 = ((std_img+1).astype(np.float)/2.0*255).astype(np.uint8)
                PIL.Image.fromarray((img_w_joints*255).astype(np.uint8)).save(vis_filename)
                # PLOT
                #print(img_w_joints.shape)
                #plt.imshow(img_w_joints)
                #plt.show()
                vis_filename=train_dir + "vis/crop_BTFM_%08d.png" % i
                img_w_joints=plot_lsp_joints(src_img, joints, proc_para=proc_para)
                PIL.Image.fromarray((img_w_joints*255).astype(np.uint8)).save(vis_filename)

            
            take_notes("LSP %04d TRAIN %08d\n" % (i+SKIP+1, train_id), "./data_log.txt")
            train_id += 1
            count_work += 1
            counter[cur_sample['set']] = counter[cur_sample['set']]+1

    print("work ratio = %f, (%d / %d)" 
          % (count_work/count_all, count_work, count_all))
    print(counter)
    return train_id, test_id

def check_joints(joints, vformat, dims):
    """
    joints: LSP format
    vformat: which dataset this came from (to interpret v correctly)
    dims: width, height
    """
    joints_ok = True

    if vformat == 'LSP':
        joints_ok = check_lsp_joints(joints, dims)
    elif vformat == 'LSPET':
        joints_ok = check_lspet_joints(joints, dims)
    elif vformat == 'MPII':
        joints_ok = check_mpii_joints(joints, dims)
    elif vformat == 'COCO':
        joints_ok = check_coco_joints(joints, dims)
    elif vformat == '3DPW':
        joints_ok = check_3dpw_joints(joints, dims)
    elif vformat == 'MI3':
        joints_ok = check_mi3_joints(joints, dims)

    return joints_ok

def check_lsp_joints(joints, dims):
    joints_ok=False
    jx=joints[0,:]
    jy=joints[1,:]
    xinimage=(jx>0)&(jx<dims[0])
    yinimage=(jy>0)&(jy<dims[1])
    validjoints=xinimage & yinimage
    if np.sum(validjoints) >= MIN_JOINTS_NEEDED:
        joints_ok=True
    return joints_ok

def check_lspet_joints(joints, dims):
    joints_ok=False
    jx=joints[0,:]
    jy=joints[1,:]
    xinimage=(jx>0)&(jx<dims[0])
    yinimage=(jy>0)&(jy<dims[1])
    validjoints=xinimage & yinimage
    if np.sum(validjoints) >= MIN_JOINTS_NEEDED:
        joints_ok=True
    return joints_ok

def check_mpii_joints(joints, dims):
    joints_ok=False
    jx=joints[0,:]
    jy=joints[1,:]
    xinimage=(jx>0)&(jx<dims[0])
    yinimage=(jy>0)&(jy<dims[1])
    validjoints=xinimage & yinimage
    if np.sum(validjoints) >= MIN_JOINTS_NEEDED:
        joints_ok=True
    return joints_ok

def check_coco_joints(joints, dims):
    joints_ok=False
    jx=joints[0,:]
    jy=joints[1,:]
    xinimage=(jx>0)&(jx<dims[0])
    yinimage=(jy>0)&(jy<dims[1])
    validjoints=xinimage & yinimage
    if np.sum(validjoints) >= MIN_JOINTS_NEEDED_COCO:
        joints_ok=True
    return joints_ok

def check_3dpw_joints(joints, dims):
    joints_ok=False
    jx=joints[0,:]
    jy=joints[1,:]
    xinimage=(jx>0)&(jx<dims[0])
    yinimage=(jy>0)&(jy<dims[1])
    validjoints=xinimage & yinimage
    if np.sum(validjoints) >= MIN_JOINTS_NEEDED_3DPW:
        joints_ok=True
    return joints_ok

def check_mi3_joints(joints, dims):
    joints_ok=False
    jx=joints[0,:]
    jy=joints[1,:]
    xinimage=(jx>0)&(jx<dims[0])
    yinimage=(jy>0)&(jy<dims[1])
    validjoints=xinimage & yinimage
    if np.sum(validjoints) >= MIN_JOINTS_NEEDED_MI3:
        joints_ok=True
    return joints_ok

# Multiprocessing code
def mp_load_entries(q, doneq, btfm_json, base_dir, total_to_process):
    # Load dataset
    with open(btfm_json, 'rb') as infile:
        btfm_dataset=json.load(infile)
    
    num_btfm_samples=len(btfm_dataset)-SKIP
    doneq.put({'cmd': 'init', 'length': num_btfm_samples}, True)
    total_to_process.value = num_btfm_samples

    for i in range(num_btfm_samples):
        cur_sample=btfm_dataset[i+SKIP]
        joints=get_joints_lspformat(cur_sample)

        # read sil
        try:
            src_gt_sil = np.array(PIL.Image.open(base_dir + cur_sample['silhouette']))
        except KeyError as e:
            doneq.put({'cmd':'skip', 'info': cur_sample}, True) # Block as long as needed
            continue

        if cur_sample['set'] in ['LSPET', 'MPII', 'MI3']:
            # LSPET, MPII encode silhouette as 3-channel but only need 1
            # MI3 has foreground/background info in Red channel (channel 0)
            src_gt_sil = src_gt_sil[:,:,0]

        # Make sure there are enough valid joints
        img = PIL.Image.open(base_dir + cur_sample['path'])
        width, height = img.size
        img.close()
        result = check_joints(joints, cur_sample['set'], (width, height))
        if result is False:
            doneq.put({'cmd':'skip', 'info': cur_sample}, True) # Block as long as needed
            continue

        # If we're here, this is a good sample. Enqueue for HMR PP
        q.put({'cmd':'sample', 'info': cur_sample, 'sil': src_gt_sil, 'joints': joints}, True) # Block as long as needed

    # All done. Let process exit
    pass

def mp_record_stats(q,train_id,test_id,num_processed):
    running = True

    # Create a counter to keep track of stats per set
    counter={}
    counter['LSP']=0
    counter['LSPET']=0
    counter['MPII']=0
    counter['COCO']=0
    counter['3DPW']=0
    counter['MI3']=0
    counter['SSP']=0

    while running:
        try:
            qcmd = q.get(True, 5) # 5 second timeout
            if qcmd['cmd'] == 'quit':
                pbar.refresh()
                pbar=None  # Clear variable so we don't get extra print
                running = False
            elif qcmd['cmd'] == 'init':  # Set the total number of items to process
                total_items=qcmd['length']
                pbar = tqdm(total=total_items)
            elif qcmd['cmd'] == 'done':  # Signal item successfully processed
                cur_sample=qcmd['info']
                counter[cur_sample['set']] = counter[cur_sample['set']]+1
                take_notes("BTFM %08d KEEP\n" % (cur_sample['ID']), NOTES_FILE)
                pbar.update(1)
                num_processed.value += 1
            elif qcmd['cmd'] == 'skip':  # Signal item skipped
                cur_sample=qcmd['info']
                take_notes("BTFM %08d DROP\n" % (cur_sample['ID']), NOTES_FILE)
                pbar.update(1)
                num_processed.value += 1
        except Empty:
            pass # Just restart loop, not a problem

    # Display stats at end
    used_items = sum([counter[x] for x in counter.keys()])
    print(counter)
    print('Used items: ' + str(used_items))
    print('Total items: ' + str(total_items))
    print('Work ratio: ' + str(used_items/total_items))
    train_id.value=used_items

def mp_hmrpp(sampleq,hmrq,base_dir):
    running = True

    while running:
        try:
            qcmd = sampleq.get(True, 5) # 5 second timeout
            if qcmd['cmd'] == 'quit':
                running = False
                hmrq.put({'cmd':'quit'}, True) # Forward to next block
                sampleq.put({'cmd':'quit'}, True) # Put it back for others
            elif qcmd['cmd'] == 'sample':  # Process a sample through HMR PP
                cur_sample=qcmd['info']
                src_gt_sil=qcmd['sil']
                ori_img = np.array(PIL.Image.open(base_dir + cur_sample['path']))
                # Get data ready for HMR prediction
                if cur_sample['set'] == 'LSP':
                    std_img, proc_para = predict_pponly(ori_img)
                elif cur_sample['set'] == 'MI3':
                    bbox=np.array(cur_sample['bbox'])
                    std_img, proc_para = predict_pponly(ori_img, 
                                                        use_j_bbox=True, 
                                                        j_bbox=bbox)
                else:
                    std_img, proc_para = predict_pponly(ori_img, 
                                                        True, 
                                                        src_gt_sil)

                # Enqueue for HMR
                hmrq.put({'cmd':'predict', 'info':cur_sample,
                          'std_img': std_img, 'proc_para': proc_para,
                          'sil': src_gt_sil,'joints': qcmd['joints']}, True)

        except Empty:
            pass # Just restart loop, not a problem

    # Anything to do on exit?
    pass

def mp_hmr(hmrq,hmrdoneq):
    running = True

    hmr_pred = hmr_predictor()

    while running:
        try:
            qcmd = hmrq.get(True, 5) # 5 second timeout
            if qcmd['cmd'] == 'quit':
                running = False
                hmrdoneq.put({'cmd':'quit'}, True) # Forward to next block
            elif qcmd['cmd'] == 'predict':  # Do HMR inference
                # Do HMR prediction and enqueue results
                cur_sample=qcmd['info']
                std_img=qcmd['std_img']
                proc_para=qcmd['proc_para']

                verts, cam = hmr_pred.predict_nopp(std_img, proc_para)

                hmrdoneq.put({'cmd':'hmr_post', 'info':cur_sample,
                   'sil': qcmd['sil'],'joints': qcmd['joints'],
                   'std_img':std_img, 'proc_para':proc_para,
                   'verts':verts, 'cam': cam}, True)
        except Empty:
            pass # Just restart loop, not a problem

    # Anything to do on exit?
    pass

def mp_hmrpost(hmrdoneq,doneq,train_dir):
    running = True

    faces = np.load("../predef/smpl_faces.npy")
    face_num = len(faces)
    
    renderer = rd.SMPLRenderer(face_path = 
                               "../predef/smpl_faces.npy")
    
    with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
        mesh_joint = pickle.load(fp)


    while running:
        try:
            qcmd = hmrdoneq.get(True, 5) # 5 second timeout
            if qcmd['cmd'] == 'quit':
                running = False
                doneq.put({'cmd':'quit'}, True) # Tell next block to stop
                hmrdoneq.put({'cmd':'quit'}, True) # Tell other workers to stop
            elif qcmd['cmd'] == 'hmr_post':  # Do HMR post-processing
                cur_sample=qcmd['info']
                std_img = qcmd['std_img']
                src_gt_sil = qcmd['sil']
                joints = qcmd['joints']
                proc_para = qcmd['proc_para']
                verts = qcmd['verts']
                cam = qcmd['cam']

                src_img = ((std_img+1).astype(np.float)/2.0*255).astype(np.uint8)

                # save img
                img_file = "img/BTFM_%08d.png" % (cur_sample['ID'])
                PIL.Image.fromarray(src_img).save(train_dir + img_file)

                # process sil
                gt_sil = proc_sil(src_gt_sil, proc_para)

                # get proj sil
                proj_sil = renderer.silhouette(verts = verts,
                                               cam = cam,
                                               img_size = src_img.shape,
                                               norm = False)

                # make TriMesh
                mesh = make_trimesh(verts, faces, compute_vn = True)
                vert_norms = mesh.vertex_normals()

                # get joint move
                unseen_mode = True if (cur_sample['set'] in ['LSPET','COCO','3DPW']) else False
                new_jv, _, joint_move, joint_posi = get_joint_move(verts, 
                                                       joints, 
                                                       proc_para,
                                                       mesh_joint,
                                                       unseen_mode=unseen_mode)
                joint_move = joint_move.flatten()

                # joint deform
                fd_ja = fast_deform_dja(weight = 10.0)
                ja_verts = fd_ja.deform(np.asarray(verts), new_jv)

                # get achr move
                proj_sil_ja = renderer.silhouette(verts = ja_verts,
                                               norm = False)

                _, achr_verts, achr_move = get_achr_move(gt_sil, 
                                                         ja_verts, 
                                                         vert_norms,
                                                         proj_sil_ja)
                achr_posi = get_anchor_posi(achr_verts)
                
                # save sil
                compose_sil = np.stack((gt_sil, proj_sil, proj_sil_ja))
                compose_sil = np.moveaxis(compose_sil, 0, 2)
                compose_sil = PIL.Image.fromarray(compose_sil.astype(np.uint8))
                compose_sil.save(train_dir + "sil/%08d.png" % cur_sample['ID'])

                # save para
                proc_para['end_pt'] = proc_para['end_pt'].tolist()
                proc_para['start_pt'] = proc_para['start_pt'].tolist()
                para = {"verts": verts.tolist(),
                        "vert_norms": vert_norms.tolist(),
                        "proc_para": proc_para,
                        "joint_move": joint_move.tolist(),
                        "joint_posi": joint_posi.tolist(),
                        "achr_move": achr_move.tolist(),
                        "achr_posi": achr_posi.tolist(),
                        "img_file": img_file,
                       }
                with open(train_dir + "para/%08d.json" % cur_sample['ID'], 'wb') as fp:
                    json.dump(para, fp)

                # Report sample done
                doneq.put({'cmd':'done', 'info':cur_sample}, True)
                
        except Empty:
            pass # Just restart loop, not a problem

    # Anything to do on exit?
    pass

def proc_btfm_parallel(train_dir, test_dir, train_id, test_id, btfm_json, base_dir, hmr_pred, visualize_joints=False, preview=False):
    
    # Set up multiprocessing communication objects
    sampleq = Queue(1000) # Limit size because this queue can fill much faster than others
    doneq = Queue()
    hmrq = Queue()
    hmrdoneq = Queue()

    # Values filled in by stats process. Might need to update since there
    # are other numbers that are more useful.
    train_id = Value('i', 0)
    test_id = Value('i', 0)
    total_to_process = Value('i', 0)
    num_processed = Value('i', 0)

    # Create processes
    p_stats=Process(target=mp_record_stats, args=(doneq,train_id,test_id,num_processed))
    p_load=Process(target=mp_load_entries, args=(sampleq, doneq, btfm_json, base_dir, total_to_process))
    hmrpp_list=[Process(target=mp_hmrpp, args=(sampleq, hmrq, base_dir)) for x in range(NUM_HMRPP_WORKERS)]
    p_hmr=Process(target=mp_hmr, args=(hmrq, hmrdoneq))
    hmrpost_list=[Process(target=mp_hmrpost, args=(hmrdoneq, doneq, train_dir)) for x in range(NUM_HMRPOST_WORKERS)]

    # Start processes - start p_load last since it will start filling up
    # subsequent queues and kick everything off
    p_hmr.start()
    sleep(10)  # Give HMR a few seconds to get set up (just to make output nice-looking)
    # Also, something bad happens if p_hmr isn't started first and some samples get dropped
    p_stats.start()
    [x.start() for x in hmrpost_list]
    [x.start() for x in hmrpp_list]
    p_load.start()

    # Wait for entire DB to be read and fed into processing pipeline
    p_load.join()
    while True:
        sleep(1)
        queuestats = [x.qsize() for x in [sampleq, hmrq, hmrdoneq, doneq]]
        #print('Queues: ' + str(queuestats))
        #print('Num processed: ' + str(num_processed.value) + '/' + str(total_to_process.value))
        if num_processed.value == total_to_process.value:
            break
    #print('Num processed (last): ' + str(num_processed.value))
    # Then, send cancel signal through pipeline
    sampleq.put({'cmd':'quit'}, True)
    [x.join() for x in hmrpp_list]
    p_hmr.join()
    [x.join() for x in hmrpost_list]
    p_stats.join()

    ## make train set
    #tr = trange(num_btfm_samples-SKIP, desc='Bar desc', leave=True)
    #for i in tr:
    #    tr.set_description("BTFM - train part")
    #    tr.refresh() # to show immediately the update
    #    sleep(0.0001)  # Used to be 0.01
    #    
    #    count_all += 1

    #    cur_sample=btfm_dataset[i+SKIP]
    #    joints=get_joints_lspformat(cur_sample)

    #    # read sil
    #    try:
    #        src_gt_sil = np.array(PIL.Image.open(
    #                 base_dir + cur_sample['silhouette']))
    #    except KeyError as e:
    #        continue

    #    if cur_sample['set'] in ['LSPET', 'MPII', 'MI3']:
    #        # LSPET, MPII encode silhouette as 3-channel but only need 1
    #        # MI3 has foreground/background info in Red channel (channel 0)
    #        src_gt_sil = src_gt_sil[:,:,0]

    #    # Make sure there are enough valid joints
    #    img = PIL.Image.open(base_dir + cur_sample['path'])
    #    width, height = img.size
    #    img.close()
    #    result = check_joints(joints, cur_sample['set'], (width, height))
    #    if result is False:
    #        continue
    #    
    #    # TODO: judge using filter. Will need to do something here or judge during BTFM dataset creation
    #    #result = lsp_filter(lsp_joints[:,:,i+SKIP], src_gt_sil)
    #    #if result is False:
    #    #    take_notes("LSP %04d BAN -1\n" % (i+SKIP+1), "./data_log.txt")
    #    #    continue
    #    
    #    # read ori img
    #    ori_img = np.array(PIL.Image.open(
    #              base_dir + cur_sample['path']))

    #    # hmr predict
    #    # TODO: MI3 will need a different bbox/silhouette
    #    if cur_sample['set'] == 'LSP':
    #        verts, cam, proc_para, std_img = hmr_pred.predict(ori_img)
    #    elif cur_sample['set'] == 'MI3':
    #        bbox=np.array(cur_sample['bbox'])
    #        verts, cam, proc_para, std_img = hmr_pred.predict(ori_img, 
    #                                                          use_j_bbox=True, 
    #                                                          j_bbox=bbox)
    #    else:
    #        verts, cam, proc_para, std_img = hmr_pred.predict(ori_img, 
    #                                                          True, 
    #                                                          src_gt_sil)
    #    
    #    # unnormalize std_img
    #    src_img = ((std_img+1).astype(np.float)/2.0*255).astype(np.uint8)
    #    
    #    # save img (used to be 1-based, shouldn't matter)
    #    img_file = "img/BTFM_%08d.png" % (i+SKIP)
    #    PIL.Image.fromarray(src_img).save(train_dir + img_file)
    #    
    #    # process sil
    #    #print('src_gt_sil: ' + str(src_gt_sil.shape) + ' ' + cur_sample['set'])
    #    gt_sil = proc_sil(src_gt_sil, proc_para)

    #    # get proj sil
    #    proj_sil = renderer.silhouette(verts = verts,
    #                                   cam = cam,
    #                                   img_size = src_img.shape,
    #                                   norm = False)

    #    # make TriMesh
    #    mesh = make_trimesh(verts, faces, compute_vn = True)
    #    vert_norms = mesh.vertex_normals()

    #    # get joint move
    #    unseen_mode = True if (cur_sample['set'] in ['LSPET','COCO','3DPW']) else False
    #    new_jv, _, joint_move, joint_posi = get_joint_move(verts, 
    #                                           joints, 
    #                                           proc_para,
    #                                           mesh_joint,
    #                                           unseen_mode=unseen_mode)
    #    joint_move = joint_move.flatten()

    #    # joint deform
    #    fd_ja = fast_deform_dja(weight = 10.0)
    #    ja_verts = fd_ja.deform(np.asarray(verts), new_jv)

    #    # get achr move
    #    proj_sil_ja = renderer.silhouette(verts = ja_verts,
    #                                   norm = False)

    #    _, achr_verts, achr_move = get_achr_move(gt_sil, 
    #                                             ja_verts, 
    #                                             vert_norms,
    #                                             proj_sil_ja)
    #    achr_posi = get_anchor_posi(achr_verts)
    #    
    #    # save sil
    #    compose_sil = np.stack((gt_sil, proj_sil, proj_sil_ja))
    #    compose_sil = np.moveaxis(compose_sil, 0, 2)
    #    compose_sil = PIL.Image.fromarray(compose_sil.astype(np.uint8))
    #    compose_sil.save(train_dir + "sil/%08d.png" % train_id)

    #    # save para
    #    proc_para['end_pt'] = proc_para['end_pt'].tolist()
    #    proc_para['start_pt'] = proc_para['start_pt'].tolist()
    #    para = {"verts": verts.tolist(),
    #            "vert_norms": vert_norms.tolist(),
    #            "proc_para": proc_para,
    #            "joint_move": joint_move.tolist(),
    #            "joint_posi": joint_posi.tolist(),
    #            "achr_move": achr_move.tolist(),
    #            "achr_posi": achr_posi.tolist(),
    #            "img_file": img_file,
    #           }
    #    with open(train_dir + "para/%08d.json" % train_id, 'wb') as fp:
    #        json.dump(para, fp)

    #    if visualize_joints:
    #        #vis_filename=train_dir + "vis/BTFM_%08d.png" % train_id
    #        vis_filename=train_dir + "vis/BTFM_%08d.png" % i
    #        img_w_joints=plot_lsp_joints(ori_img, joints)
    #        #img255 = ((std_img+1).astype(np.float)/2.0*255).astype(np.uint8)
    #        PIL.Image.fromarray((img_w_joints*255).astype(np.uint8)).save(vis_filename)
    #        # PLOT
    #        #print(img_w_joints.shape)
    #        #plt.imshow(img_w_joints)
    #        #plt.show()
    #        vis_filename=train_dir + "vis/crop_BTFM_%08d.png" % i
    #        img_w_joints=plot_lsp_joints(src_img, joints, proc_para=proc_para)
    #        PIL.Image.fromarray((img_w_joints*255).astype(np.uint8)).save(vis_filename)

    #    
    #    take_notes("LSP %04d TRAIN %08d\n" % (i+SKIP+1, train_id), "./data_log.txt")
    #    train_id += 1
    #    count_work += 1
    #    counter[cur_sample['set']] = counter[cur_sample['set']]+1

    return train_id.value, test_id.value

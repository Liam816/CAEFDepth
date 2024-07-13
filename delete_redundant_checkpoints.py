import numpy as np
import os
import glob
import copy

# print('hello')
# exit()

use_runInfo = True
use_manual_mode = True


if use_runInfo is False:
    root = "./checkpoints/FasterNet-S_EdgeNetV2_DecoderV7_06-29_3"
    path = os.path.join(root, '*.pth')
    print('path', path)
    all_checkpoints_list = glob.glob(path)

    training_metrics = np.load(os.path.join(root, 'training_metrics.npy'), allow_pickle=True)

    rmse = training_metrics[0, :]
    mae = training_metrics[1, :]
    delta1 = training_metrics[2, :]
    delta2 = training_metrics[3, :]
    delta3 = training_metrics[4, :]
    absrel = training_metrics[5, :]
    lg10 = training_metrics[6, :]
    gpu_time = training_metrics[7, :]

    preserved_checkpoints_list = []
    preserved_checkpoints_dict = {}

    rmse_min_loc = np.where(rmse == np.min(rmse))
    print("rmse_min_loc:{}".format(rmse_min_loc))
    preserved_checkpoints_list.append('checkpoint_{}.pth'.format(rmse_min_loc[0][0]))
    preserved_checkpoints_dict[rmse_min_loc[0][0]] = 'checkpoint_{}.pth'.format(rmse_min_loc[0][0])

    # mae_max_loc = np.where(mae == np.max(mae))
    # print("mae_max_loc:{}".format(mae_max_loc))

    delta1_max_loc = np.where(delta1 == np.max(delta1))
    print("delta1_max_loc:{}".format(delta1_max_loc))
    preserved_checkpoints_list.append('checkpoint_{}.pth'.format(delta1_max_loc[0][0]))
    preserved_checkpoints_dict[delta1_max_loc[0][0]] = 'checkpoint_{}.pth'.format(delta1_max_loc[0][0])

    delta2_max_loc = np.where(delta2 == np.max(delta2))
    print("delta2_max_loc:{}".format(delta2_max_loc))
    preserved_checkpoints_list.append('checkpoint_{}.pth'.format(delta2_max_loc[0][0]))
    preserved_checkpoints_dict[delta2_max_loc[0][0]] = 'checkpoint_{}.pth'.format(delta2_max_loc[0][0])

    delta3_max_loc = np.where(delta3 == np.max(delta3))
    print("delta3_max_loc:{}".format(delta3_max_loc))
    preserved_checkpoints_list.append('checkpoint_{}.pth'.format(delta3_max_loc[0][0]))
    preserved_checkpoints_dict[delta3_max_loc[0][0]] = 'checkpoint_{}.pth'.format(delta3_max_loc[0][0])

    absrel_min_loc = np.where(absrel == np.min(absrel))
    print("absrel_min_loc:{}".format(absrel_min_loc))
    preserved_checkpoints_list.append('checkpoint_{}.pth'.format(absrel_min_loc[0][0]))
    preserved_checkpoints_dict[absrel_min_loc[0][0]] = 'checkpoint_{}.pth'.format(absrel_min_loc[0][0])

    lg10_min_loc = np.where(lg10 == np.min(lg10))
    print("lg10_min_loc:{}".format(lg10_min_loc))
    preserved_checkpoints_list.append('checkpoint_{}.pth'.format(lg10_min_loc[0][0]))
    preserved_checkpoints_dict[lg10_min_loc[0][0]] = 'checkpoint_{}.pth'.format(lg10_min_loc[0][0])

    # print(preserved_checkpoints_list)
    # print(preserved_checkpoints_dict)

    for key in preserved_checkpoints_dict:
        preserved = os.path.join(root, 'checkpoint_{}.pth'.format(key))
        print("preserved:", preserved)
        all_checkpoints_list.remove(preserved)

    # print(all_checkpoints_list)

    for i in range(len(all_checkpoints_list)):
        print('deleting:{}'.format(all_checkpoints_list[i]))
        os.remove(all_checkpoints_list[i])


else:
    # rmse = []
    # mae = []
    # delta1 = []
    # delta2 = []
    # delta3 = []
    # absrel = []
    # lg10 = []
    # gpu_time = []
    #
    # log_file = "runInfo_FasterNet-S_EdgeNetV3_DecoderV8_07-01.txt"
    # print("log_runInfo_file:{}".format(log_file))
    # ues_print = False
    # with open(os.path.join('.', log_file), 'r') as f:
    #     for line in f:
    #         line = line.strip()
    #         if "RMSE=" in line:
    #             if ues_print is True:
    #                 print(line)
    #             rmse.append(float(line[5:]))
    #         elif "MAE=" in line:
    #             if ues_print is True:
    #                 print(line)
    #             mae.append(float(line[4:]))
    #         elif "Delta1=0." in line:
    #             if ues_print is True:
    #                 print(line)
    #             delta1.append(float(line[7:]))
    #         elif "Delta2=0." in line:
    #             if ues_print is True:
    #                 print(line)
    #             delta2.append(float(line[7:]))
    #         elif "Delta3=0." in line:
    #             if ues_print is True:
    #                 print(line)
    #             delta3.append(float(line[7:]))
    #         elif "REL=" in line:
    #             if ues_print is True:
    #                 print(line)
    #             absrel.append(float(line[4:]))
    #         elif "Lg10=" in line:
    #             if ues_print is True:
    #                 print(line)
    #             lg10.append(float(line[5:]))
    #         elif "t_GPU=0." in line:
    #             if ues_print is True:
    #                 print(line)
    #             gpu_time.append(float(line[6:]))
    #
    # # print("rmse:{}, type:{}".format(rmse[0], type(rmse[0])))
    # # print("mae:{}, type:{}".format(mae[0], type(mae[0])))
    # # print("delta1:{}, type:{}".format(delta1[0], type(delta1[0])))
    # # print("delta2:{}, type:{}".format(delta2[0], type(delta2[0])))
    # # print("delta3:{}, type:{}".format(delta3[0], type(delta3[0])))
    # # print("absrel:{}, type:{}".format(absrel[0], type(absrel[0])))
    # # print("lg10:{}, type:{}".format(lg10[0], type(lg10[0])))
    # # print("gpu_time:{}, type:{}".format(gpu_time[0], type(gpu_time[0])))
    #
    # rmse = np.array(rmse)
    # mae = np.array(mae)
    # delta1 = np.array(delta1)
    # delta2 = np.array(delta2)
    # delta3 = np.array(delta3)
    # absrel = np.array(absrel)
    # lg10 = np.array(lg10)
    # gpu_time = np.array(gpu_time)
    #
    # metrics_dict = {}
    #
    # rmse_min_loc = np.where(rmse == np.min(rmse))  # rmse_min_loc是一个包含了numpy数组的tuple
    # # print("rmse_min_loc:{}".format(rmse_min_loc[0]))
    # metrics_dict["rmse"] = rmse_min_loc[0].tolist()
    # # print("rmse_min_loc:{}, type:{}".format(rmse_min_loc, type(rmse_min_loc)))
    # # exit()
    #
    # mae_min_loc = np.where(mae == np.min(mae))
    # # print("mae_min_loc:{}".format(mae_min_loc[0]))
    # # metrics_dict["mae"] = mae_min_loc[0].tolist()
    #
    # delta1_max_loc = np.where(delta1 == np.max(delta1))
    # # print("delta1_max_loc:{}".format(delta1_max_loc[0]))
    # metrics_dict["delta1"] = delta1_max_loc[0].tolist()
    #
    # delta2_max_loc = np.where(delta2 == np.max(delta2))
    # # print("delta2_max_loc:{}".format(delta2_max_loc[0]))
    # metrics_dict["delta2"] = delta2_max_loc[0].tolist()
    #
    # delta3_max_loc = np.where(delta3 == np.max(delta3))
    # # print("delta3_max_loc:{}".format(delta3_max_loc[0]))
    # metrics_dict["delta3"] = delta3_max_loc[0].tolist()
    #
    # absrel_min_loc = np.where(absrel == np.min(absrel))
    # # print("absrel_min_loc:{}".format(absrel_min_loc[0]))
    # metrics_dict["absrel"] = absrel_min_loc[0].tolist()
    #
    # lg10_min_loc = np.where(lg10 == np.min(lg10))
    # # print("lg10_min_loc:{}".format(lg10_min_loc[0]))
    # metrics_dict["lg10"] = lg10_min_loc[0].tolist()
    #
    # gpu_time_min_loc = np.where(gpu_time == np.min(gpu_time))
    # # print("gpu_time_min_loc:{}".format(gpu_time_min_loc[0]))
    # # metrics_dict["gpu_time"] = gpu_time_min_loc[0].tolist()
    #
    # for k, v in metrics_dict.items():
    #     print("k:{}, v:{}".format(k, v))
    #
    # res_dict = {}  # key: checkpoint序号 value: 对应的metric名字
    # # target = "rmse"
    # # for i in range(len(metrics_dict["rmse"])):
    # #     candidate = metrics_dict["rmse"][i]
    # #     res_dict[candidate] = ["rmse"]
    # #     # print("res_dict:{}".format(res_dict))
    # #
    # #     # temp_dict = metrics_dict.copy()
    # #     temp_dict = copy.deepcopy(metrics_dict)  # 深拷贝
    # #     temp_dict.pop("rmse")
    # #
    # #     for k, v in temp_dict.items():
    # #         for j in range(len(v)):
    # #             if candidate == v[j]:
    # #                 res_dict[candidate].append(k)
    # #                 metrics_dict[k].remove(v[j])
    #
    # # target_list = ["rmse", "mae", "delta1", "delta2", "delta3", "absrel", "lg10", "gpu_time"]
    # target_list = ["rmse", "delta1", "delta2", "delta3", "absrel", "lg10"]
    # for target in target_list:
    #     for i in range(len(metrics_dict[target])):
    #         candidate = metrics_dict[target][i]
    #         res_dict[candidate] = [target]
    #
    #         temp_dict = copy.deepcopy(metrics_dict)  # 深拷贝
    #         temp_dict.pop(target)
    #
    #         for k, v in temp_dict.items():
    #             for j in range(len(v)):
    #                 if candidate == v[j]:
    #                     res_dict[candidate].append(k)
    #                     metrics_dict[k].remove(v[j])
    #
    # # for k, v in metrics_dict.items():
    # #     print("metrics_dict k:{}, v:{}".format(k, v))
    #
    # for k, v in res_dict.items():
    #     # print("res_dict k:{}, v:{}".format(k, v))
    #     print("ckpt_{} counts:{} metrics:{}".format(k, len(v), v))
    #
    # exit()




    log_file = 'runInfo_MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-16.txt'
    print('log_file: {}'.format(log_file))

    if 'kitti' not in log_file:
        rmse = []
        mae = []
        delta1 = []
        delta2 = []
        delta3 = []
        absrel = []
        lg10 = []
        gpu_time = []

        # NOTE: 在NYU上训练的模型
        ues_print = False
        with open(os.path.join('.', log_file), 'r') as f:
            for line in f:
                line = line.strip()
                if "RMSE=" in line:
                    if ues_print is True:
                        print(line)
                    rmse.append(float(line[5:]))
                elif "MAE=" in line:
                    if ues_print is True:
                        print(line)
                    mae.append(float(line[4:]))
                elif "Delta1=0." in line:
                    if ues_print is True:
                        print(line)
                    delta1.append(float(line[7:]))
                elif "Delta2=0." in line:
                    if ues_print is True:
                        print(line)
                    delta2.append(float(line[7:]))
                elif "Delta3=0." in line:
                    if ues_print is True:
                        print(line)
                    delta3.append(float(line[7:]))
                elif "REL=" in line and 'Sq' not in line:
                    if ues_print is True:
                        print(line)
                    absrel.append(float(line[4:]))
                elif "Lg10=" in line:
                    if ues_print is True:
                        print(line)
                    lg10.append(float(line[5:]))
                elif "t_GPU=0." in line:
                    if ues_print is True:
                        print(line)
                    gpu_time.append(float(line[6:]))


        ckpt_nums = 60
        rmse = np.array(rmse[:ckpt_nums])
        absrel = np.array(absrel[:ckpt_nums])
        lg10 = np.array(lg10[:ckpt_nums])
        delta1 = np.array(delta1[:ckpt_nums])
        delta2 = np.array(delta2[:ckpt_nums])
        delta3 = np.array(delta3[:ckpt_nums])
        mae = np.array(mae[:ckpt_nums])
        gpu_time = np.array(gpu_time[:ckpt_nums])

        res_dict = dict()
        res_dict['rmse'] = rmse
        res_dict['absrel'] = absrel
        res_dict['lg10'] = lg10
        res_dict['delta1'] = delta1
        res_dict['delta2'] = delta2
        res_dict['delta3'] = delta3

        sorted_dict = {}
        for k, v in res_dict.items():
            arr_unique = np.unique(v)
            # print('arr_unique"{}'.format(arr_unique))
            if 'delta' in k:  # 准确率 需要倒序从大到小排列 最大的排第一 表示效果最好
                sorted_indices = np.argsort(arr_unique)[::-1] + 1
            else:  # 误差项 默认从小到大排列 最小的排第一 表示效果最好
                sorted_indices = np.argsort(arr_unique) + 1
            # print('sorted_indices"{}'.format(sorted_indices))
            sorted_dict[k] = (arr_unique, sorted_indices)

        ckpt_dict = {}
        for i in range(len(rmse)):  # len(rmse)  有n个ckpt
            temp_list = []
            for k, v in sorted_dict.items():  # 有6个指标
                # print('k:{}----------'.format(k))
                for t in range(len(v[0])):  # 有m个候选值
                    # print('res_dict[k][i]:{} v[0][t]:{}'.format(res_dict[k][i], v[0][t]))
                    if res_dict[k][i] == v[0][t]:
                        # print('v[0][t]:{}'.format(v[0][t]))
                        temp_list.append(v[1][t])
            # ckpt_dict['ckpt_' + str(i)] = temp_list
            ckpt_dict['ckpt_' + str(i)] = (temp_list, temp_list.count(1), sum(temp_list) / 6)

        # for k, v in ckpt_dict.items():
        #     print('k:{} v:{} nums:{} v_avg:{}'.format(k, v, v.count(1), sum(v) / 6))
        #     # print('k:{} v:{}'.format(k, v))
        # # exit()

        sorted_ckpt = sorted(ckpt_dict.items(), key=lambda x: x[1][2])
        for i in range(len(sorted_ckpt)):
            print('{}: {}'.format(sorted_ckpt[i][0], sorted_ckpt[i][1]))
        # exit()

        # NOTE: 手动指定删除
        # ckpt_path = log_file.split('.')[0].split('runInfo_')[-1]
        ckpt_path = log_file.replace('runInfo_', '').replace('.txt', '')
        root = os.path.join('./checkpoints', ckpt_path)
        path = os.path.join(root, '*.pth')
        print('working path: {}'.format(path))
        all_checkpoints_list = glob.glob(path)
        # print('all_checkpoints_list:\n', all_checkpoints_list)
        # exit()

        # preserved_ckp_list = [33, 30, 25, 38]
        preserved_ckp_list = [35, 32, 30, 39]

        for i in range(len(preserved_ckp_list)):
            preserved = os.path.join(root, 'checkpoint_{}.pth'.format(preserved_ckp_list[i]))
            # preserved = os.path.join(root, 'ckpt_{}.pth'.format(preserved_ckp_list[i]))
            print("preserved:", preserved)
            all_checkpoints_list.remove(preserved)

        for i in range(len(all_checkpoints_list)):
            print('deleting:{}'.format(all_checkpoints_list[i]))
            os.remove(all_checkpoints_list[i])

        # NOTE: 自动删除
        # ckpt_path = log_file.split('.')[0].split('runInfo_')[-1]
        # root = os.path.join('./checkpoints', ckpt_path)
        # # root = "./checkpoints/DDRNet-23-slim_pre_LiamEdge_DecoderKS_06-15"
        # path = os.path.join(root, '*.pth')
        # print('path', path)
        # all_checkpoints_list = glob.glob(path)
        # print('all_checkpoints_list:\n', all_checkpoints_list)

        # preserved_nums = 4
        # for i in range(preserved_nums):
        #     preserved = os.path.join(root, 'checkpoint_{}.pth'.format(sorted_ckpt[i][0].split('_')[-1]))
        #     print("preserved:", preserved)
        #     all_checkpoints_list.remove(preserved)
        # exit()
        #
        # for i in range(len(all_checkpoints_list)):
        #     if os.path.exists(all_checkpoints_list[i]):
        #         print('deleting:{}'.format(all_checkpoints_list[i]))
        #         os.remove(all_checkpoints_list[i])
        # exit()

    else:
        Delta1 = []
        Delta2 = []
        Delta3 = []
        AbsRel = []
        SqRel = []
        RMSE = []

        ues_print = False  # True False
        with open(os.path.join('.', log_file), 'r') as f:
            for line in f:
                line = line.strip()
                if "rmse:" in line:
                    if ues_print is True:
                        print(line)
                    RMSE.append(float(line[8:]))
                elif "sq_rel:" in line:
                    if ues_print is True:
                        print(line)
                    SqRel.append(float(line[10:]))
                elif "abs_rel:" in line:
                    if ues_print is True:
                        print(line)
                    AbsRel.append(float(line[11:]))
                elif "delta1:" in line:
                    if ues_print is True:
                        print(line)
                    Delta1.append(float(line[10:]))
                elif "delta2:" in line:
                    if ues_print is True:
                        print(line)
                    Delta2.append(float(line[10:]))
                elif "delta3:" in line:
                    if ues_print is True:
                        print(line)
                    Delta3.append(float(line[10:]))

        # exit()

        rmse = np.array(RMSE[:])
        sqrel = np.array(SqRel[:])
        absrel = np.array(AbsRel[:])
        delta1 = np.array(Delta1[:])
        delta2 = np.array(Delta2[:])
        delta3 = np.array(Delta3[:])

        res_dict = dict()
        res_dict['rmse'] = rmse
        res_dict['sqrel'] = sqrel
        res_dict['absrel'] = absrel
        res_dict['delta1'] = delta1
        res_dict['delta2'] = delta2
        res_dict['delta3'] = delta3

        sorted_dict = {}
        for k, v in res_dict.items():
            arr_unique = np.unique(v)
            if 'delta' in k:  # 准确率 需要倒序从大到小排列 最大的排第一 表示效果最好
                sorted_indices = np.argsort(arr_unique)[::-1] + 1
            else:  # 误差项 默认从小到大排列 最小的排第一 表示效果最好
                sorted_indices = np.argsort(arr_unique) + 1
            sorted_dict[k] = (arr_unique, sorted_indices)

        ckpt_dict = {}
        for i in range(len(rmse)):  # len(rmse)  有n个ckpt
            temp_list = []
            for k, v in sorted_dict.items():  # 有6个指标
                for t in range(len(v[0])):  # 有m个候选值
                    if res_dict[k][i] == v[0][t]:
                        temp_list.append(v[1][t])
            ckpt_dict['ckpt_' + str(i)] = (temp_list, temp_list.count(1), sum(temp_list) / 6)

        sorted_ckpt = sorted(ckpt_dict.items(), key=lambda x: x[1][2])
        for i in range(len(sorted_ckpt)):
            print('{}: {}'.format(sorted_ckpt[i][0], sorted_ckpt[i][1]))

        # exit()

        # NOTE: 手动指定删除
        ckpt_path = log_file.replace('runInfo_', '').replace('.txt', '')
        root = os.path.join('./checkpoints', ckpt_path)
        path = os.path.join(root, '*.pth')
        print('working path: {}'.format(path))
        all_checkpoints_list = glob.glob(path)

        preserved_ckp_list = [39]

        for i in range(len(preserved_ckp_list)):
            preserved = os.path.join(root, 'checkpoint_{}.pth'.format(preserved_ckp_list[i]))
            print("preserved:", preserved)
            all_checkpoints_list.remove(preserved)

        for i in range(len(all_checkpoints_list)):
            print('deleting:{}'.format(all_checkpoints_list[i]))
            os.remove(all_checkpoints_list[i])

        # NOTE: 自动删除
        # ckpt_path = log_file.split('.')[0].split('runInfo_')[-1]
        # root = os.path.join('./checkpoints', ckpt_path)
        # # root = "./checkpoints/DDRNet-23-slim_pre_LiamEdge_DecoderKS_06-15"
        # path = os.path.join(root, '*.pth')
        # print('path', path)
        # all_checkpoints_list = glob.glob(path)
        # print('all_checkpoints_list:\n', all_checkpoints_list)

        # preserved_nums = 4
        # for i in range(preserved_nums):
        #     preserved = os.path.join(root, 'checkpoint_{}.pth'.format(sorted_ckpt[i][0].split('_')[-1]))
        #     print("preserved:", preserved)
        #     all_checkpoints_list.remove(preserved)
        # exit()
        #
        # for i in range(len(all_checkpoints_list)):
        #     if os.path.exists(all_checkpoints_list[i]):
        #         print('deleting:{}'.format(all_checkpoints_list[i]))
        #         os.remove(all_checkpoints_list[i])
        # exit()




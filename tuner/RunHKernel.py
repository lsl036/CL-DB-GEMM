from dataclasses import field
import datetime
import subprocess
import time
import numpy as np
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
# from GPyopt import constrains

# FOR GEMM kernel tune first 
# Bayesian domains 
GEMM_bds = [{'name': 'GEMMK', 'type': 'discrete', 'domain': [0]},            #0
            {'name': 'KREG', 'type': 'discrete', 'domain': [1]},          #1
            {'name': 'KWG', 'type': 'discrete', 'domain': [16, 32]},                #2
            {'name': 'KWI', 'type': 'discrete', 'domain': [2]},              #3
            {'name': 'MDIMA', 'type': 'discrete', 'domain': [8, 16, 32]},   #4
            {'name': 'MDIMC', 'type': 'discrete', 'domain': [8, 16, 32]},       #5
            {'name': 'MWG', 'type': 'discrete', 'domain': [16, 32, 64, 128]},   #6
            {'name': 'NDIMB', 'type': 'discrete', 'domain': [8, 16, 32]},   #7
            {'name': 'NDIMC', 'type': 'discrete', 'domain': [8, 16, 32]},       #8
            {'name': 'NWG', 'type': 'discrete', 'domain': [16, 32, 64, 128]},   #9
            {'name': 'SA', 'type': 'discrete', 'domain': [0, 1]},               #10
            {'name': 'SB', 'type': 'discrete', 'domain': [0, 1]},               #11
            {'name': 'STRM', 'type': 'discrete', 'domain': [0, 1]},             #12
            {'name': 'STRN', 'type': 'discrete', 'domain': [0, 1]},             #13
            {'name': 'VWM', 'type': 'discrete', 'domain': [1, 2, 4, 8]},        #14
            {'name': 'VWN', 'type': 'discrete', 'domain': [1, 2, 4, 8]}]        #15

# 参数的限制条件包括 :
# Requirement for unrolling the KWG loop 
#                    MultipleOfX, {"KWG", "KWI"}, 
# Required for integer MWI and NWI
#                    MultipleOfXMulY, {"MWG", "MDIMC", "VWM"}
#                    MultipleOfXMulY, {"NWG", "NDIMC", "VWN"}
# Required for integer MWIA and NWIB
#                    MultipleOfXMulY, {"MWG", "MDIMA", "VWM"}
#                    MultipleOfXMulY, {"NWG", "NDIMB", "VWN"}
# When GEMMK == 0, KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
#                   MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "MDIMA"}
#                   MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "NDIMB"}
#     extra         IsEqual, {"MDIMC", "MDIMA"}
#     extra         IsEqual, {"NDIMC", "NDIMB"}
#     extra         IsEqual, {"SA", "SB"}
# When GEMMK == 1,
#                   MultipleOfX, {"KREG", "VWN"}
#     extra         IsEqual, {"MDIMC", "MDIMA"}
#     extra         IsEqual, {"NDIMC", "NDIMB"}
GEMM_constraints = [{'name':'KWG_is_multiple_KWI', 'constraint':'x[:,2]%x[:,3]-0.1'},
                    {'name':'MWG_is_multiple_MDIMCVWM', 'constraint':'x[:,6]%(x[:,5]*x[:,14])-0.1'},
                    {'name':'NWG_is_multiple_NDIMCVWN', 'constraint':'x[:,9]%(x[:,8]*x[:,15])-0.1'},
                    {'name':'MWG_is_multiple_MDIMAVWM', 'constraint':'x[:,6]%(x[:,4]*x[:,14])-0.1'},
                    {'name':'NWG_is_multiple_NDIMBVWN', 'constraint':'x[:,9]%(x[:,7]*x[:,15])-0.1'},
                    {'name':'MDIMA_is_multiple', 'constraint':'(x[:,2]%(x[:,5]*x[:,8]/x[:,4])-0.1)*(x[:,0]==0)'},
                    {'name':'NDIMB_is_multiple', 'constraint':'(x[:,2]%(x[:,5]*x[:,8]/x[:,7])-0.1)*(x[:,0]==0)'},
                    {'name':'KREG_is_multiple_VWN', 'constraint':'(x[:,1]%x[:,15]-0.1)*(x[:,0]==1)'}]
#  还有几个是 GEMMK == 1 的时候需要开启的extra限制条件，可以考虑先单纯测试GEMMK == 0 的情况的
#  {'name':'MDIMC_euqal_MDIMA', 'constraint':'(x[:,5]==x[:,4])*(x[:,0]==1)'}
#  {'name':'NDIMC_euqal_NDIMB', 'constraint':'(x[:,8]==x[:,7])*(x[:,0]==1)'}
# 

# Optimization objective 优化对象
def Run_Kernel_with_Params(parameters):
    cmd = "./clblast_sample_Bayesian_Tuner -precision 32 -GEMMK %s -KREG %s -KWG %s -KWI %s -MDIMA %s -MDIMC %s -MWG %s -NDIMB %s -NDIMC %s -NWG %s -SA %s -SB %s -STRM %s -STRN %s -VWM %s -VWN %s"

    parameters = parameters[0]

    # 以程序执行时间来打分
    start_time = time.time()
    # 需要写个执行程序
    os.system(cmd % ( parameters[0], parameters[1], parameters[2], parameters[3], parameters[4],
                      parameters[5], parameters[6], parameters[7], parameters[8], parameters[9],
                      parameters[10], parameters[11], parameters[12], parameters[13], parameters[14],
                      parameters[15] ))
    # for i in range(0,16) :
    #     print("P[%d] = %d"%(i, parameters[i]))
    # print("** GEMMK == %d **"%(parameters[0]))
    # print("P[2] = %d"%(parameters[2]))
    # print("P[5] = %d"%(parameters[5]))
    # print("P[7] = %d"%(parameters[7]))
    # print("P[8] = %d"%(parameters[8]))
    # print("P[4] = %d"%(parameters[4]))
    # print("MWG % (MDIMC*VWM)  = ", parameters[6]%(parameters[5]*parameters[14]))
    # print("MWG % (MDIMA*VWM)  = ", parameters[6]%(parameters[4]*parameters[14]))
    # print("NWG % (NDIMC*VWN)  = ", parameters[9]%(parameters[8]*parameters[15]))
    # print("NWG % (NDIMB*VWN)  = ", parameters[9]%(parameters[7]*parameters[15]))
    # print("Judge A = ", parameters[2]%(parameters[5]*parameters[8]/parameters[4]))
    # print("Judge B = ", parameters[2]%(parameters[5]*parameters[8]/parameters[7]))
    # print("P[1] = %d"%(parameters[1]))
    # print("P[15] = %d"%(parameters[1]))
    # print("KREG %% VWN = %d"%(parameters[1]%parameters[15]))
    
    end_time = time.time()
    score = end_time - start_time

    score = np.array(score)
    return score

def parse_output(output):
    lines = output.split('\n')
    last_line = lines[-1]
    # print("last_line = ",last_line)
    
    fields = last_line.split('|')
    needed_result = fields[-3].strip()
    return needed_result


def Run_Xgemm_Kernel(parameters):
    parameters = parameters[0]
    try:
        # Half precision
        cmd = ['./clblast_sample_BO_Tuner_half', '-platform', '2', '-precision', '16', '-GEMMK', str(parameters[0]), '-KREG', str(parameters[1]), '-KWG', str(parameters[2]), '-KWI', str(parameters[3]), '-MDIMA', str(parameters[4]), '-MDIMC', str(parameters[5]), '-MWG', str(parameters[6]), '-NDIMB', str(parameters[7]), '-NDIMC', str(parameters[8]), '-NWG', str(parameters[9]), '-SA', str(parameters[10]), '-SB', str(parameters[11]), '-STRM', str(parameters[12]), '-STRN', str(parameters[13]), '-VWM', str(parameters[14]), '-VWN', str(parameters[15])]
        
        # 使用subprocess.Popen运行C++可执行程序
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 等待可执行程序运行完成并捕获输出
        stdout, stderr = process.communicate()

        # 解码输出结果（默认为字节类型）
        output = stdout.decode().strip()

        # 检查执行状态
        returncode = process.returncode
        if returncode == 0:
            # print("C++ executable finished successfully.")
            print("Output:")
            print(output)
            ret_result = parse_output(output)
            if ret_result == '-':
                score = 0.0
            else:
                score = float(ret_result)
            # print(type(score))
            print("score = ",score)
        else:
            print("C++ executable exited with an error (return code {}):".format(returncode))
            print(stderr.decode().strip())  # 输出错误信息
        return score
        
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the C++ executable:", e)
        return None

if __name__ == "__main__":
    optimizer = BayesianOptimization(f=Run_Xgemm_Kernel, 
                                 domain=GEMM_bds,
                                 constraints = GEMM_constraints,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True, 
                                 maximize=True)
    
    res_name = 'Xgemm_Bayesian'
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%M%S')
    res_name = res_name + mkfile_time + '.pdf'

    iteration_num = 50
    optimizer.run_optimization(max_iter=iteration_num, verbosity=True)

    # 查看优化结果
    print("Optimized GFLOPS:", -optimizer.fx_opt)  # 最优性能指标(GFLOPS)

    # 绘制优化结果的收敛图
    optimizer.plot_convergence()

    # 调整子图之间的布局，避免纵坐标标签重叠
    plt.tight_layout()

    # 显示图形
    plt.show()
    plt.legend();
    plt.savefig(res_name)
    # p_test = [0, 1, 32, 2, 8, 8, 32, 16, 16, 64, 1, 1, 0, 0, 4, 4]
    # print(type(p_test))
    # Run_Xgemm_Kernel(p_test)

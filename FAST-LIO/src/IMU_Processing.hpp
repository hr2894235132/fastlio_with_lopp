#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (20) \
// typedef pcl::PointXYZINormal PointType;

const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };

/// *************IMU Process and undistortion
class ImuProcess {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();

    ~ImuProcess();

    void Reset();

    void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);

    void set_extrinsic(const V3D &transl, const M3D &rot);

    void set_extrinsic(const V3D &transl);

    void set_extrinsic(const MD(4, 4) &T);

    void set_gyr_cov(const V3D &scaler);

    void set_acc_cov(const V3D &scaler);

    void set_gyr_bias_cov(const V3D &b_g);

    void set_acc_bias_cov(const V3D &b_a);

    Eigen::Matrix<double, 12, 12> Q;

    void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                 PointCloudXYZI::Ptr pcl_un_);

    ofstream fout_imu;
    // typedef Vector3d V3D;
    V3D cov_acc;
    V3D cov_gyr;
    V3D cov_acc_scale;
    V3D cov_gyr_scale;
    V3D cov_bias_gyr;
    V3D cov_bias_acc;
    double first_lidar_time;

private:
    void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);

    void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                      PointCloudXYZI &pcl_in_out);

    PointCloudXYZI::Ptr cur_pcl_un_;
    sensor_msgs::ImuConstPtr last_imu_;
    deque<sensor_msgs::ImuConstPtr> v_imu_;
    vector<Pose6D> IMUpose; // (时间，加速度，角速度，速度，位置，旋转矩阵）
    vector<M3D> v_rot_pcl_;
    M3D Lidar_R_wrt_IMU;
    V3D Lidar_T_wrt_IMU;
    V3D mean_acc; // 初始化时得到的加速度均值
    V3D mean_gyr; // 初始化时得到的角速度均值
    V3D angvel_last;
    V3D acc_s_last;
    double start_timestamp_;
    double last_lidar_end_time_;
    int init_iter_num = 1; // 初始化时，imu数据迭代的次数
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
};

ImuProcess::ImuProcess()
        : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1) {
    init_iter_num = 1;
    Q = process_noise_cov();
    cov_acc = V3D(0.1, 0.1, 0.1);
    cov_gyr = V3D(0.1, 0.1, 0.1);
    cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc = V3D(0.0001, 0.0001, 0.0001);
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    Lidar_T_wrt_IMU = Zero3d;
    Lidar_R_wrt_IMU = Eye3d;
    last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

/* 重置 */
void ImuProcess::Reset() {
    // ROS_WARN("Reset ImuProcess");
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    imu_need_init_ = true;
    start_timestamp_ = -1;
    init_iter_num = 1;
    v_imu_.clear();
    IMUpose.clear();
    last_imu_.reset(new sensor_msgs::Imu());
    cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4, 4) &T) // #define MD(a,b) Matrix<double, (a), (b)>
{
    Lidar_T_wrt_IMU = T.block<3, 1>(0, 3);
    Lidar_R_wrt_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl) // Vector3d V3D
{
    Lidar_T_wrt_IMU = transl;
    Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot) {
    Lidar_T_wrt_IMU = transl; // T:3*1 vector
    Lidar_R_wrt_IMU = rot; // R:3*3 matrix
}

void ImuProcess::set_gyr_cov(const V3D &scaler) // 陀螺仪
{
    cov_gyr_scale = scaler; // 3*1 vector
}

void ImuProcess::set_acc_cov(const V3D &scaler) // 加速度计
{
    cov_acc_scale = scaler; // 3*1 vector
}

// 噪声
void ImuProcess::set_gyr_bias_cov(const V3D &b_g) {
    cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a) {
    cov_bias_acc = b_a;
}

/** 1. 初始化重力、陀螺仪偏差、加速度计和陀螺仪协方差
   ** 2. 将加速度测量值归化为单位重力 **/
// 前20帧静置初始化，静置时的加速度均值，作为imu的重力；静置时的角速度均值作为重力的bias，将imu重力的xyz互相相乘作为重力的协方差矩阵
// 将imu状态量加入kf的状态列表中
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N) {
    /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity **/

    V3D cur_acc, cur_gyr;

    if (b_first_frame_) // 第一帧
    {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration; // 线性加速度
        const auto &gyr_acc = meas.imu.front()->angular_velocity; // 角速度
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z; // 记录第一帧加速度为加速度均值
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z; // 记录第一帧角速度为角速度均值
        first_lidar_time = meas.lidar_beg_time; // 记录lidar起始时间（绝对）
    }

    // 计算方差
    // 读取所有imu信息
    for (const auto &imu: meas.imu) {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += (cur_acc - mean_acc) / N; // 更新加速度均值
        mean_gyr += (cur_gyr - mean_gyr) / N; // 更新角速度均值

        // 更新加速度、角速度协方差
        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) /
                                            (N * N);// cwiseProduct：输出相同位置的两个矩阵中各个系数的乘积所组成的矩阵
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

        // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

        N++;
    }
    state_ikfom init_state = kf_state.get_x(); // 初始状态
    init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2); // -acc * (g / acc)，负值，根据加速度均值，重力模长做归一化，重力记为S2流形
    const auto &orientation = meas.imu.front()->orientation;
    // 如果orientation消息可用的话,以IMU的orientaion作为初始姿态
    if (orientation.x != 0 || orientation.y != 0 || orientation.z != 0 || orientation.w != 0) {
        Eigen::Quaterniond qua(orientation.w, orientation.x, orientation.y, orientation.z);
        init_state.rot = qua;
    }
    //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
    init_state.bg = mean_gyr;
    init_state.offset_T_L_I = Lidar_T_wrt_IMU;
    init_state.offset_R_L_I = Lidar_R_wrt_IMU;
    kf_state.change_x(init_state);

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P(); // 获取协方差矩阵模版
    init_P.setIdentity(); // 初始化协方差
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001; // 外参r
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001; // 外参t
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001; // bias g
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001; // bias a
    init_P(21, 21) = init_P(22, 22) = 0.00001; // S2
    kf_state.change_P(init_P);
    last_imu_ = meas.imu.back();

}

/*** 去畸变 ***/
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                              PointCloudXYZI &pcl_out) {
    /*** add the imu of the last frame-tail to the imu of current frame-head ***/
    /*** 将最后一帧尾部的imu添加到当前帧头部的imu ***/
    auto v_imu = meas.imu; // imu数据序列
    v_imu.push_front(last_imu_); // 把上一个imu数据插入到头
    const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_beg_time;
    const double &pcl_end_time = meas.lidar_end_time;

    /*** sort point clouds by offset time ***/
    // 从小到大按时间排序
    // 按之前已设定的曲率（单点时间戳）对点云进行排列
    pcl_out = *(meas.lidar);
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
//  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
    // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

    /*** Initialize IMU pose ***/
    // imu initial pose
    state_ikfom imu_state = kf_state.get_x();
    IMUpose.clear();
    //保存数据：时间差，加速度，角速度，线速度，位置，姿态（后三个为推断值）
    //设定初始时刻相对状态(相对于imu积分初始状态的时间，上一加速度，上一角速度，速度，位置，旋转矩阵）
    IMUpose.push_back(
            set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

    /*** forward propagation at each imu point ***/
    /*** 前向传播 ***/
    V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    M3D R_imu;

    double dt = 0;

    input_ikfom in; // 输入状态量：记录加速度和角速度
    // 离散中值法 前向传播
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        // 选取能cover住lidar时间戳的imu数据帧
        if (tail->header.stamp.toSec() < last_lidar_end_time_) continue;

        // j与j+1的加速度和角速度均值
        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " "
                 << acc_avr.transpose() << endl;

        // 加速度去除重力影响
        acc_avr = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

        // 如果IMU开始时刻早于上次雷达最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
        if (head->header.stamp.toSec() < last_lidar_end_time_) {
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
            // dt = tail->header.stamp.toSec() - pcl_beg_time;
        } else {
            // 两个IMU时刻之间的时间间隔
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }

        // 协方差矩阵
        in.acc = acc_avr;
        in.gyro = angvel_avr;
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
        // 每收到一次imu数据帧则进行一次前向更新（先验）(两imu数据间隔时间差，协方差矩阵，输入数据）
        kf_state.predict(dt, Q, in);

        /* save the poses at each IMU measurements */
        /* 保存每次IMU测量的位姿 */
        imu_state = kf_state.get_x();
        angvel_last = angvel_avr - imu_state.bg;
        acc_s_last = imu_state.rot * (acc_avr - imu_state.ba);
        for (int i = 0; i < 3; i++) {
            acc_s_last[i] += imu_state.grav[i]; // 加上重力得到世界坐标系的加速度
        }
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time; // 后一个IMU时刻距离此次雷达开始的时间间隔
        // 保存帧内imu数据 m+1 时刻的pose、前一时刻与当前时刻imu数据的均值（w系）、v、p、r、
        IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos,
                                     imu_state.rot.toRotationMatrix()));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    /*** 计算帧末的位置和姿态预测 ***/
    // 判断雷达结束时间是否晚于IMU
    // 计算雷达末尾姿态
    // 最后一个IMU时刻可能早于雷达末尾 也可能晚于雷达末尾
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time); // lidar终点与imu终点时间差
    // 确保lidar与imu的处理时间对齐
    kf_state.predict(dt, Q, in); // in 此时为最后一个imu数据，传播到 lidar终点时刻 与 imu终点时刻 较大者

    imu_state = kf_state.get_x(); // lidar终点时刻与imu终点时刻较大者，imu位姿
    last_imu_ = meas.imu.back(); // 记录最后一个imu数据为上一imu数据，用于下一次imu前向传播的开头
    last_lidar_end_time_ = pcl_end_time; // 记录点云结束时间为上一帧结束时间

    /*** undistort each lidar point (backward propagation) ***/
    /*** 反向传播 去畸变 ***/
    if (pcl_out.points.begin() == pcl_out.points.end()) return;
    auto it_pcl = pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) { // traversal from end to begin
        auto head = it_kp - 1;
        auto tail = it_kp;
        // 位姿、线速度使用head的imu数据，
        //j-1时刻imu的p、v、r
        R_imu << MAT_FROM_ARRAY(head->rot);
        // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        // 加速度、角速度使用tail的数据
        // j时刻的加速度和角速度
        acc_imu << VEC_FROM_ARRAY(tail->acc);
        angvel_avr << VEC_FROM_ARRAY(tail->gyr);

        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - head->offset_time; // 相对于前一imu时刻的时间差(单点距imu帧头的时间差)

            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
            /* 变换到“结束”帧，仅使用旋转
             * 注意：补偿方向与帧的移动方向相反
             * 所以如果我们想补偿时间戳i到帧e的一个点
             * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)  其中T_ei在全局框架中表示 */
            // 按时间戳的差值进行插值
            M3D R_i(R_imu * Exp(angvel_avr, dt));

            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z); // lidar系下索引i点的坐标
            // T_ei 从imu终点位置指向索引i时刻imu位置的平移向量，w系, 即T_i - T_e
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
            V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i *
                                                                                                  (imu_state.offset_R_L_I *
                                                                                                   P_i +
                                                                                                   imu_state.offset_T_L_I) +
                                                                                                  T_ei) -
                                                                     imu_state.offset_T_L_I);// not accurate! // .conjugate()共轭矩阵

            // save Undistorted points and their rotation
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_out.points.begin()) break;
        }
    }
}

void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         PointCloudXYZI::Ptr cur_pcl_un_) {
    double t1, t2, t3;
    t1 = omp_get_wtime();

    if (meas.imu.empty()) { return; };
    ROS_ASSERT(meas.lidar != nullptr); // 检验条件是否成功,若失败则终端程序并输出文件/行数/条件

    if (imu_need_init_) {
        /// The very first lidar frame
        IMU_init(meas, kf_state, init_iter_num);

        imu_need_init_ = true;

        last_imu_ = meas.imu.back();

        state_ikfom imu_state = kf_state.get_x();
        if (init_iter_num > MAX_INI_COUNT) { // MAX_INI_COUNT 20
            cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
            imu_need_init_ = false;

            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;
            ROS_INFO("IMU Initial Done");
            // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
            fout_imu.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
        }

        return;
    }

    UndistortPcl(meas, kf_state, *cur_pcl_un_);

    t2 = omp_get_wtime();
    t3 = omp_get_wtime();

    // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}

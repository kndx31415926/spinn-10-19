function [t_hit, v_hit, w0, log] = spinn_mechanical_armNN(params25, opts)
% NN 学生链（稳定版）：几何最速方向(+抗重力混合) + 直接“扭矩缩放”的双顶帽功率限幅 + 方向保护 + 可选硬限位
% 关键改动：
%   1) 不再用 p/omega 回算扭矩；改为直接缩放扭矩满足“先分轴、后总功率”的上限（对齐 PID 基线）。
%   2) 增强起步稳定性：dir_gravity_mix、sign_guard、bootstrap(肩轴最小占比)。
%
% 输入
%   params25 : 1x25
%              [m1 m2 m3, dq0(3), damp1 zeta1 damp2 zeta2 damp3 zeta3,
%               init_deg(3), tgt_deg(3), dtheta(3), Pmax, Prated(3)]
%   opts     : 可选，见 defaults()。
%
% 输出
%   t_hit, v_hit : 命中时间/命中瞬时末端速度；未命中为 NaN
%   w0           : t=0 的 NN 份额（投影+下界后）
%   log          : 记录（t,p(3xK),v(1xK),q(3xK),dq(3xK),Pmax,Prated,hit,radius,w(Kx3)）

    if nargin<2, opts=struct(); end
    o = defaults(); fn = fieldnames(o);
    for i=1:numel(fn), if ~isfield(opts,fn{i}), opts.(fn{i}) = o.(fn{i}); end, end

    % ---- 载入模型（缺失则回退等分）----
    hasModel = true;
    modelFile = 'trained_model_spinn.mat';
    if ~isfile(modelFile), modelFile = 'trained_model.mat'; end
    if isfile(modelFile)
        S = load(modelFile);    % 需要 trainedNet, muX, sigmaX, muY, sigmaY
        need = {'trainedNet','muX','sigmaX','muY','sigmaY'};
        if ~all(isfield(S,need)), hasModel=false; warning('[spinn_mechanical_armNN] 模型字段不全，回退均分 w。'); end
    else
        hasModel=false; warning('[spinn_mechanical_armNN] 未找到模型，回退均分 w。'); S=struct();
    end

    % ---- 解包 25 维 ----
    m  = params25(1:3);
    dq = params25(4:6);
    damp = [params25(7) params25(9) params25(11)];
    zeta = [params25(8) params25(10) params25(12)];
    q0_deg = params25(13:15);
    qT_deg = params25(16:18);
    Pmax   = params25(22);
    Prated = params25(23:25);

    if opts.force_zero_init
        dq = [0 0 0]; q0_deg = [0 0 0];
    end

    % ---- 常量、初值 ----
    g=9.81; L=[0.24,0.214,0.324];
    q  = deg2rad(q0_deg(:));
    dq = dq(:);
    qT = deg2rad(qT_deg(:));

    % t 轴
    K = floor(opts.t_final/opts.dt)+1;  t = (0:K-1)*opts.dt;

    % 记录
    p_hist = zeros(3,K);
    v_hist = zeros(1,K);
    q_hist = zeros(3,K); q_hist(:,1)=q;
    dq_hist= zeros(3,K); dq_hist(:,1)=dq;
    w_hist = zeros(K,3);

    % 先算一次性 w0（用于显示）
    w0 = predict_w(q, dq);

    % OU 噪声 & PID 积分态（仅 use_pid=true 时使用）
    xi=zeros(3,1); eI=zeros(3,1); ePrev=zeros(3,1);

    hit=false; t_hit=NaN; v_hit=NaN;

    % ===== 主循环 =====
    for k=2:K
        % --- 方向：几何最速 or PID ---
        if ~opts.use_pid
            [xE,yE] = fk_end(q,L); [xT,yT] = fk_end(qT,L);
            r  = [xT-xE; yT-yE]; nr = norm(r); if nr>0, r=r/nr; else, r=[0;0]; end
            J  = jacobian_planar(q,L);
            dir_geo = J.' * r;                           % 几何方向
            % 抗重力混合：dir = Jᵀ r_hat + α * unit(+G(q))
            [Mq, Cq, Gq] = computeDynamics(m(1),m(2),m(3),L(1),L(2),L(3),g, q, dq);
            if opts.dir_gravity_mix > 0
                g_unit = Gq / max(norm(Gq), 1e-9);       % +G(q) 单位方向
                dir_raw = dir_geo + opts.dir_gravity_mix * g_unit;
            else
                dir_raw = dir_geo;
            end
            nd = norm(dir_raw); if nd>0, dir_raw = dir_raw/nd; end
            tau_des = opts.k_dir_max * dir_raw;
        else
            e = qT - q; eI = eI + e*opts.dt; eD = (e - ePrev)/opts.dt; ePrev = e;
            tau_des = opts.pid.Kp(:).*e + opts.pid.Ki(:).*eI + opts.pid.Kd(:).*eD;
            [Mq, Cq, Gq] = computeDynamics(m(1),m(2),m(3),L(1),L(2),L(3),g, q, dq);
        end

        % --- 角速度下界（仅用于功率计算；不做 p/ω 回算） ---
        om = dq;
        om_nz = om;
        mask = ~isfinite(om_nz) | (abs(om_nz) < opts.omega_eps);
        sgn = sign(tau_des); sgn(sgn==0) = 1;
        om_nz(mask) = opts.omega_eps .* sgn(mask);

        % --- 早期 bootstrap 覆盖 NN 权重（比例/均分），并强制肩轴最小占比 ---
        if opts.online_recompute_w
            w_now = predict_w(q, dq);
        else
            w_now = w0;
        end
        if k <= opts.bootstrap_steps
            switch lower(opts.bootstrap_mode)
                case 'ratio'
                    p_seed = abs(tau_des(:) .* om_nz(:));
                    s = sum(p_seed); if s>0, w_now = (p_seed(:).'/s); else, w_now = [1 1 1]/3; end
                case 'equal'
                    w_now = [1 1 1]/3;
            end
        end
        % 地板与肩轴占比
        w_now(1) = max(w_now(1), max(opts.w_floor, opts.shoulder_min));
        w_now = max(w_now, opts.w_floor); w_now = w_now / sum(w_now);
        w_hist(k,:) = w_now;

        % --- fill-power：把“扭矩种子”按需要放大一些（不是放大 p 再倒回 τ）---
        if opts.fill_power
            s_now = sum(abs(tau_des(:) .* om_nz(:)));
            if s_now > 0 && s_now < opts.fill_alpha * Pmax
                tau_des = tau_des * min(opts.p_boost_max, (opts.fill_alpha * Pmax)/s_now);
            end
        end

        % --- 双顶帽功率限幅（核心：直接缩放“扭矩”） ---
        % 1) 分轴功率上限：min(w_i*Pmax_eff, Prated_i)
        Pmax_eff = Pmax;
        if isfield(opts,'Pmax_gate') && isa(opts.Pmax_gate,'function_handle')
            Pmax_eff = max(0, min(Pmax, opts.Pmax_gate(norm([xT-xE;yT-yE]), Pmax, k, t(k))));
        end
        cap_i = min(w_now(:)*Pmax_eff, Prated(:));        % 3x1

        % 2) 计算限幅前轴功率
        p_raw = abs(tau_des(:) .* om_nz(:));              % 3x1

        % 3) 先按“分轴功率上限”逐轴缩放扭矩
        tau_lim = tau_des(:).';
        for iAx = 1:3
            if p_raw(iAx) > (cap_i(iAx) + 1e-12)
                scale_i = cap_i(iAx) / (p_raw(iAx) + eps);
                tau_lim(iAx) = tau_lim(iAx) * scale_i;
            end
        end

        % 4) 再按“总功率上限”整体缩放扭矩
        p_after_axis = abs(tau_lim(:) .* om_nz(:));
        P_after_axis = sum(p_after_axis);
        if P_after_axis > Pmax_eff && P_after_axis > 0
            tau_lim = tau_lim * (Pmax_eff / P_after_axis);
        end

        % 5) 可选：扭矩绝对上限（额外保险）
        if isfield(opts,'tau_rated') && ~isempty(opts.tau_rated)
            tau_lim = sign(tau_lim) .* min(abs(tau_lim), opts.tau_rated(:).');
        end

        % --- 方向一致性保护（绝不朝反） ---
        if opts.sign_guard
            dotdir = (tau_lim(:).' * dir_geo);
            if dotdir < 0
                tau_lim = -tau_lim;
            end
        end

        % --- 动力学推进（加微正则，消除奇异警告） ---
        xi = xi + (-xi/opts.ou_tau)*opts.dt + sqrt(2*opts.dt/opts.ou_tau)*randn(3,1);
        D  = diag(max(0, damp(:).*(1 + zeta(:).*xi)));

        ddq = (Mq + 1e-6*eye(3)) \ (tau_lim(:) - Cq*dq - Gq - D*dq);
        dq  = dq + ddq*opts.dt;
        q   = q  + dq *opts.dt;

        % 硬限位（可选）
        if isstruct(opts.joint) && ~isempty(opts.joint)
            [q, dq] = enforce_joint_hard_limits(q, dq, opts.joint);
        end

        % 记录
        p_hist(:,k) = abs(tau_lim(:) .* dq);          % 记录限幅后功率（带号或绝对值均可；这里记绝对值）
        q_hist(:,k) = q;
        dq_hist(:,k)= dq;

        vE = jacobian_planar(q,L)*dq; 
        v_hist(k) = norm(vE);

        % 命中判定
        [xE,yE] = fk_end(q,L); 
        if ~hit && hypot(xE-xT,yE-yT) <= opts.radius
            hit=true; t_hit=t(k); v_hit=v_hist(k);
            break;
        end
    end

    k_end = k;
    log = struct('t',t(1:k_end), 'p',p_hist(:,1:k_end), 'v',v_hist(1:k_end), ...
                 'q',q_hist(:,1:k_end), 'dq',dq_hist(:,1:k_end), ...
                 'Pmax',Pmax, 'Prated',Prated, 'hit',hit, ...
                 'radius',opts.radius, 'controller','nn', 'w',w_hist(1:k_end,:));
end

% ---------- 默认参数 ----------
function o = defaults()
    o.dt = 0.002; o.t_final = 5.0; o.radius = 0.2;
    o.ou_tau = 0.30; o.omega_eps = 1e-3;
    o.k_dir_max = 25;
    o.use_pid = false;
    o.online_recompute_w = true;
    o.force_zero_init = false;

    % 关键稳态项
    o.w_floor = 0.05;                 % NN 输出最小份额
    o.fill_power = true;              % 以“扭矩种子”为对象的填充
    o.fill_alpha = 0.9;
    o.p_boost_max = 5.0;

    % 新增稳定项
    o.dir_gravity_mix = 0.35;         % [0~1] 抗重力混合
    o.sign_guard = true;              % 方向一致性保护
    o.bootstrap_steps = 180;          % 前 N 步覆盖 NN 权重
    o.bootstrap_mode  = 'ratio';      % 'ratio' | 'equal'
    o.shoulder_min    = 0.18;         % 肩轴最小占比（≥ w_floor）
    o.Pmax_gate       = [];           % 可选：近目标收功率门控
    o.tau_rated       = [];           % 可选：每轴扭矩上限（N·m）

    o.pid = struct('Kp',[60 60 60],'Ki',[0.20 0.20 0.20],'Kd',[0.10 0.10 0.10]);
    o.joint = [];                     % 可选：关节硬限位
end

% ---------- 内嵌：预测 w 并做“地板+归一” ----------
function w = predict_w(q_now, dq_now)
    persistent S_cache hasModel_cache
    % 直接用外层的 S/hasModel；这里写健壮兜底
    try
        S_local = evalin('caller','S'); has_local = evalin('caller','hasModel');
        if ~isempty(S_local), S_cache = S_local; hasModel_cache = has_local; end
    catch, end
    if isempty(hasModel_cache) || ~hasModel_cache
        w = [1 1 1]/3; return;
    end
    try
        qT_deg = evalin('caller','qT_deg'); Pmax = evalin('caller','Pmax'); Prated = evalin('caller','Prated');
        params25 = evalin('caller','params25');
        init_deg_now = rad2deg(q_now(:)).';
        dth_deg_now  = qT_deg - init_deg_now;
        x = [ params25(1:3), ...                % m1..3
              dq_now(:).', ...                  % 把当前 dq 当作 dq0
              params25(7:12), ...               % damp/zeta
              init_deg_now, qT_deg, dth_deg_now, ...
              Pmax, Prated ];
        x_norm = (x - S_cache.muX)./S_cache.sigmaX;
        y_pred = predict(S_cache.trainedNet, x_norm);
        y_pred = y_pred .* S_cache.sigmaY + S_cache.muY;
        w = y_pred(1:3);
    catch
        w = [1 1 1]/3;
    end
    % 地板 + 归一
    wf = evalin('caller','opts'); wf = wf.w_floor;
    w = max(w, wf); w = w / sum(w);
end

% ---------- 几何/工具 ----------
function [x3,y3] = fk_end(q,L)
    x1=L(1)*cos(q(1)); y1=L(1)*sin(q(1));
    x2=x1+L(2)*cos(q(1)+q(2)); y2=y1+L(2)*sin(q(1)+q(2));
    x3=x2+L(3)*cos(q(1)+q(2)+q(3)); y3=y2+L(3)*sin(q(1)+q(2)+q(3));
end

function J = jacobian_planar(q,L)
    q1=q(1); q2=q(2); q3=q(3);
    J11=-L(1)*sin(q1)-L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3);
    J12=-L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3);
    J13=-L(3)*sin(q1+q2+q3);
    J21= L(1)*cos(q1)+L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3);
    J22= L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3);
    J23= L(3)*cos(q1+q2+q3);
    J=[J11 J12 J13; J21 J22 J23];
end

% 关节硬限位：卡边→角度钳制+置零该轴速度
function [q_clamp, dq_clamp] = enforce_joint_hard_limits(q, dq, J)
    q_clamp = q; dq_clamp = dq;
    if ~isfield(J,'qmin_deg') || ~isfield(J,'qmax_deg'), return; end
    qmin = deg2rad(J.qmin_deg(:));  qmax = deg2rad(J.qmax_deg(:));
    if numel(qmin)~=3 || numel(qmax)~=3, return; end
    db   = 0.5; if isfield(J,'deadband_deg') && ~isempty(J.deadband_deg), db = J.deadband_deg; end
    db   = deg2rad(db).*ones(3,1);
    lb = qmin + db;  ub = qmax - db;

    z0 = true; if isfield(J,'zero_vel_on_contact'), z0 = logical(J.zero_vel_on_contact); end
    fr = true; if isfield(J,'freeze_inward'),       fr = logical(J.freeze_inward);       end

    for i=1:3
        if q_clamp(i) < lb(i)
            q_clamp(i) = lb(i);
            if z0, dq_clamp(i) = 0; end
            if fr && dq_clamp(i) < 0, dq_clamp(i) = 0; end
        elseif q_clamp(i) > ub(i)
            q_clamp(i) = ub(i);
            if z0, dq_clamp(i) = 0; end
            if fr && dq_clamp(i) > 0, dq_clamp(i) = 0; end
        end
    end
end

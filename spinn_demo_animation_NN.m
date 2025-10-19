function spinn_demo_animation_NN()
% 一键动画（NN 推理版，带角速度上限 + 扭矩平滑的稳态配置）

    %% 1) 参数区
    m1=0.18; m2=0.22; m3=3.40;
    dq0 = [0 0 0];
    damping = [3.2 3.2 3.2];
    zeta    = [0.010 0.010 0.010];
    tgt_deg  = [35 40 45];
    init_deg = [0 0 0];
    Pmax   = 100;
    Prated = [110 110 110];

    Kp=[60 60 60]; Ki=[0.20 0.20 0.20]; Kd=[0.10 0.10 0.10];

    cfg.dt=0.002; cfg.t_final=10; cfg.radius=0.010; cfg.omega_eps=1e-3;

    L1=0.24; L2=0.214; L3=0.324;

    %% 2) 组装 25 维
    m=[m1 m2 m3];
    dampZ=[damping(1) zeta(1) damping(2) zeta(2) damping(3) zeta(3)];
    dth_deg=tgt_deg - init_deg;
    params25=[m, dq0, dampZ, init_deg, tgt_deg, dth_deg, Pmax, Prated];

    %% 3) 学生链选项（新增稳态字段）
    opts_nn = struct( ...
        'dt',cfg.dt,'t_final',cfg.t_final,'radius',cfg.radius,'omega_eps',cfg.omega_eps, ...
        'use_pid', false, 'pid', struct('Kp',Kp,'Ki',Ki,'Kd',Kd), ...
        'online_recompute_w', true, ...
        'w_floor', 0.05, 'shoulder_min', 0.18, ...
        'fill_power', true, 'fill_alpha', 0.85, 'p_boost_max', 2.0, ...
        'dir_gravity_mix', 0.35, 'sign_guard', true, 'sign_guard_angle_deg', 95, ...
        'tau_slew', 800, 'tau_smooth_alpha', 0.40, ...
        'omega_max', [6 6 8]);

    %% 4) 调用 v1 学生链
    [t_hit, v_hit, w0, log] = spinn_mechanical_armNN(params25, opts_nn); %#ok<ASGLU>

    %% 5) 统一维度/绘图（原逻辑保持）
    t = log.t(:);

    qh_raw = log.q;
    if ndims(qh_raw)~=2, error('log.q 维度异常。'); end
    if size(qh_raw,1)==3 && size(qh_raw,2)~=3, qh = qh_raw.'; else, qh = qh_raw; end

    p_raw = log.p;
    if size(p_raw,1)==3 && size(p_raw,2)~=3, Pk3 = p_raw.'; else, Pk3 = p_raw; end

    K = min([ numel(t), size(qh,1), size(Pk3,1) ]);
    if K < 2, error('日志长度不足（K=%d）。', K); end
    t=t(1:K); qh=qh(1:K,:); Pk3=Pk3(1:K,:); Ptot = sum(abs(Pk3),2);

    xy1=zeros(K,2); xy2=zeros(K,2); xy3=zeros(K,2);
    for k=1:K
        [x1,y1,x2,y2,x3,y3] = chain_xy(qh(k,:).', L1,L2,L3);
        xy1(k,:)=[x1,y1]; xy2(k,:)=[x2,y2]; xy3(k,:)=[x3,y3];
    end
    tgt = deg2rad(tgt_deg(:));
    [xT,yT] = fk_end(tgt(1), tgt(2), tgt(3), L1,L2,L3);

    fig = figure('Name','SPINN NN Inference Demo','Color','w','Position',[60 60 1200 540]);
    try, set(fig,'Renderer','opengl'); catch, end

    ax1 = subplot(1,2,1);
    hold(ax1,'on'); axis(ax1,'equal'); grid(ax1,'on');
    R=L1+L2+L3; xlim(ax1,[-R R]); ylim(ax1,[-R R]);
    title(ax1,'3R 机械臂（NN 推理 + 双顶帽功率限幅）');
    plot(ax1, xT, yT, 'ro', 'MarkerSize',8, 'LineWidth',1.2);
    rectangle(ax1, 'Position',[xT-cfg.radius, yT-cfg.radius, 2*cfg.radius, 2*cfg.radius], ...
              'Curvature',[1,1], 'EdgeColor',[1 0 0], 'LineStyle','--', 'LineWidth',1.0);

    hPath = plot(ax1, xy3(1,1), xy3(1,2), '-', 'LineWidth',1);
    hArm  = plot(ax1, [0, xy1(1,1), xy2(1,1), xy3(1,1)], ...
                       [0, xy1(1,2), xy2(1,2), xy3(1,2)], '-o','LineWidth',3);
    hEE   = plot(ax1, xy3(1,1), xy3(1,2), 'o', 'MarkerSize',6, 'MarkerFaceColor',[0.2 0.6 1.0], 'Color','k');
    hTxt  = text(ax1, 0.02, 0.98, '', 'Units','normalized','HorizontalAlignment','left','VerticalAlignment','top','FontSize',10);

    ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'on');
    plot(ax2, t, abs(Pk3(:,1)), '-', 'LineWidth',1.2);
    plot(ax2, t, abs(Pk3(:,2)), '-', 'LineWidth',1.2);
    plot(ax2, t, abs(Pk3(:,3)), '-', 'LineWidth',1.2);
    ylabel(ax2,'|p_i| / W'); title(ax2,'三轴功率（限幅后）'); legend(ax2,'p_1','p_2','p_3');

    ax3 = subplot(2,2,4); hold(ax3,'on'); grid(ax3,'on');
    plot(ax3, t, Ptot, 'LineWidth',1.6); yline(ax3, Pmax, ':r', 'P_{max}');
    xlabel(ax3,'t / s'); ylabel(ax3,'\Sigma |p_i| / W'); title(ax3,'总功率');

    playback = 1.0; dt_est = mean(diff(t)); if ~isfinite(dt_est)||dt_est<=0, dt_est=cfg.dt; end
    stride = max(1, round(0.004 / dt_est));

    for k=1:stride:K
        if ~ishghandle(fig), return; end
        set(hArm, 'XData',[0, xy1(k,1), xy2(k,1), xy3(k,1)], ...
                  'YData',[0, xy1(k,2), xy2(k,2), xy3(k,2)]);
        set(hEE, 'XData',xy3(k,1), 'YData',xy3(k,2));
        set(hPath, 'XData',xy3(1:k,1), 'YData',xy3(1:k,2));

        w_show = w0;
        if isfield(log,'w') && size(log.w,1)>=k && all(isfinite(log.w(k,:))), w_show = log.w(k,:); end
        txt = sprintf(['t=%.3f s | hit=%d\nw=[%.2f %.2f %.2f] | P_{max}=%g W\n' ...
                       'cfg: dt=%.4f, T=%.2f, R=%.3f, \\omega_{eps}=%.1e'], ...
                      t(k), isfield(log,'hit')&&logical(log.hit), ...
                      w_show(1), w_show(2), w_show(3), Pmax, cfg.dt, cfg.t_final, cfg.radius, cfg.omega_eps);
        set(hTxt,'String',txt);
        drawnow; pause((dt_est*stride)/max(1e-6,playback));
    end

    if ~(isfield(log,'hit')&&logical(log.hit))
        warning('本次未命中（time_to_reach = NaN）。建议上调 cfg.t_final 或 cfg.radius，或增加 Pmax。');
    end
end

function [x1,y1,x2,y2,x3,y3] = chain_xy(q, L1,L2,L3)
    q1=q(1); q2=q(2); q3=q(3);
    x1=L1*cos(q1); y1=L1*sin(q1);
    x2=x1+L2*cos(q1+q2); y2=y1+L2*sin(q1+q2);
    x3=x2+L3*cos(q1+q2+q3); y3=y2+L3*sin(q1+q2+q3);
end

function [x3,y3] = fk_end(q1,q2,q3,L1,L2,L3)
    x1=L1*cos(q1); y1=L1*sin(q1);
    x2=x1+L2*cos(q1+q2); y2=y1+L2*sin(q1+q2);
    x3=x2+L3*cos(q1+q2+q3); y3=y2+L3*sin(q1+q2+q3);
end

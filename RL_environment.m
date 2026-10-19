classdef RL_environment < rl.env.MATLABEnvironment
    %AMS_RL: Custom RL environment for AMS parcel singulation.

    %% Properties
    properties
        v_treadmill = 1.8; % m/s

        d_AMS = 0.2;      % m - size of each AMS cell
        n_i_AMS = 5;      % number of AMS cells along y
        n_j_AMS = 5;      % number of AMS cells along x
        n_actions_art = 2;

        upperlim_rot = pi*45/180;
        upperlim_v = 2.2;
        lowerlim_rot = -pi*45/180;
        lowerlim_v = 0.5;

        vy_boxes = [];
        x_boxes_prec = [];
        y_boxes_prec = [];

        index_AMS = [];

        LastAction = zeros(50,1);

        i_box = [];
        j_box = [];

        n_boxes_tot_vect = [];
        n_boxes_tot = 0;
        max_gen_boxes = 15;
        cont_new_box = 0;

        d_boxes = zeros(15,1);
        x_boxes = zeros(15,1);
        y_boxes = zeros(15,1);
        x_boxes_vect = zeros(10000,15);
        y_boxes_vect = zeros(10000,15);
        cont = 0;
        pack_exited = zeros(15,1);
        exit_order = zeros(15,1);
        index_exit = 0;

        toll_contatto = 0.005;

        dt = 0.01;
        time = 0;
    end

    properties (Hidden)
        VisualizeAnimation = true
        VisualizeActions = false
        VisualizeStates = false
    end

    properties
        % 75 states = 25 AMS cells * [x, y, d]
        State = zeros(75,1)
    end

    properties(Access = protected)
        IsDone = false
    end

    properties (Transient, Access = private)
        Visualizer = []
    end

    %% Necessary Methods
    methods

        function this = RL_environment()

            ObservationInfo = rlNumericSpec([75 1]);
            ObservationInfo.Name = 'System States';
            ObservationInfo.Description = ...
                'For each of the 25 AMS cells: [x,y,d] of the parcel overlapping that cell; zeros if empty.';

            n_actions = 5*5*2;

            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name = 'AMS Action';
            ActionInfo.Description = 'r1, v1, r2, v2, ...';
            ActionInfo.LowerLimit = zeros(n_actions,1);
            ActionInfo.UpperLimit = zeros(n_actions,1);

            for ii = 1:2:n_actions
                ActionInfo.UpperLimit(ii)   = pi*45/180;
                ActionInfo.UpperLimit(ii+1) = 2.2;

                ActionInfo.LowerLimit(ii)   = -pi*45/180;
                ActionInfo.LowerLimit(ii+1) = 0.5;
            end

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

        end

        function [Observation, Reward, IsDone, Info] = step(this, Action)

            Info = [];

            this.time = this.time + this.dt;
            this.cont = this.cont + 1;

            l_AMS_matrix = this.d_AMS * this.n_j_AMS;

            ActLimUp = zeros(50,1);
            ActLimLow = zeros(50,1);

            for ii = 1:2:50
                ActLimUp(ii) = this.upperlim_rot;
                ActLimUp(ii+1) = this.upperlim_v;

                ActLimLow(ii) = this.lowerlim_rot;
                ActLimLow(ii+1) = this.lowerlim_v;
            end

            % De-normalize actions from [-1,1] to physical limits
            AMS_actions = ActLimLow + (1 + Action) .* (ActLimUp - ActLimLow) ./ 2;

            for ii = 1:50
                AMS_actions(ii) = max(ActLimLow(ii), min(ActLimUp(ii), AMS_actions(ii)));
            end

            this.LastAction = AMS_actions;

            v_AMS = zeros(5,5);
            rotation_AMS = zeros(5,5);

            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS
                    rotation_AMS(ii,jj) = AMS_actions(jj*2 - 1 + (ii-1)*10);
                    v_AMS(ii,jj)        = AMS_actions(jj*2     + (ii-1)*10);
                end
            end

            %% Generate new boxes
            if mod(round(this.time,2),2) == 0 && ...
                    this.n_boxes_tot < this.max_gen_boxes && this.time > 0.25

                this.cont_new_box = this.cont_new_box + 1;

                gen_n_boxes = 3;

                while this.n_boxes_tot + gen_n_boxes > this.max_gen_boxes
                    gen_n_boxes = gen_n_boxes - 1;
                end

                this.n_boxes_tot = this.n_boxes_tot + gen_n_boxes;
                this.n_boxes_tot_vect(this.cont_new_box) = gen_n_boxes;

                % Collect indices of the newly spawned boxes
                new_indices = (this.n_boxes_tot - gen_n_boxes + 1):this.n_boxes_tot;

                % Random initial placement for all new boxes
                for ii = new_indices
                    this.x_boxes(ii) = this.d_boxes(ii)*0.5 + ...
                        (l_AMS_matrix - this.d_boxes(ii)*0.5)*rand(1);
                    this.y_boxes(ii) = 0.001 + rand(1)*0.05;
                end

                % Iteratively resolve overlaps among all new boxes (same logic as reset)
                max_iter = 20;
                for iter = 1:max_iter
                    any_overlap = false;

                    for a = new_indices
                        for b = new_indices
                            if b <= a, continue; end

                            gap = abs(this.x_boxes(a) - this.x_boxes(b));
                            minDist = this.d_boxes(a)/2 + this.d_boxes(b)/2 + 0.1;

                            if gap < minDist
                                any_overlap = true;
                                correction = (minDist - gap) / 2;

                                if this.x_boxes(a) < this.x_boxes(b)
                                    this.x_boxes(a) = this.x_boxes(a) - correction;
                                    this.x_boxes(b) = this.x_boxes(b) + correction;
                                else
                                    this.x_boxes(a) = this.x_boxes(a) + correction;
                                    this.x_boxes(b) = this.x_boxes(b) - correction;
                                end
                            end
                        end

                        % Clamp to AMS bounds after every adjustment
                        half = this.d_boxes(a)/2;
                        if this.x_boxes(a) - half < 0
                            this.x_boxes(a) = half;
                        elseif this.x_boxes(a) + half > l_AMS_matrix
                            this.x_boxes(a) = l_AMS_matrix - half;
                        end
                    end

                    if ~any_overlap
                        break;
                    end
                end

                % Save prec values for all new boxes
                for ii = new_indices
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                end
            end

            for ii = this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = [];

            %% Kinematics and collision handling
            for ii = 1:this.n_boxes_tot

                % Wall clamp
                if this.x_boxes(ii) - this.d_boxes(ii)/2 < 0
                    this.x_boxes(ii) = this.d_boxes(ii)/2;

                elseif this.x_boxes(ii) + this.d_boxes(ii)/2 > l_AMS_matrix
                    this.x_boxes(ii) = l_AMS_matrix - this.d_boxes(ii)/2;
                end

                % Identify AMS row
                if this.y_boxes(ii) <= this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii) <= this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii) <= this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii) <= this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii) <= this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                % Identify AMS column
                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii) <= this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii) <= this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii) <= this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii) <= this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                % Box kinematics
                if this.i_box(ii) < 6

                    r = this.i_box(ii);
                    c = this.j_box(ii);

                    this.x_boxes(ii) = this.x_boxes(ii) + ...
                        v_AMS(r,c) * this.dt * sin(rotation_AMS(r,c));

                    this.y_boxes(ii) = this.y_boxes(ii) + ...
                        v_AMS(r,c) * this.dt * cos(rotation_AMS(r,c));

                    this.vy_boxes(ii) = v_AMS(r,c) * cos(rotation_AMS(r,c));

                else

                    this.y_boxes(ii) = this.y_boxes(ii) + this.v_treadmill * this.dt;
                    this.vy_boxes(ii) = this.v_treadmill;
                end

                %% Collision handling
                if ii > 1 && this.cont > 1

                    for jj = ii:-1:2

                        if abs(this.x_boxes(ii) - this.x_boxes(jj-1)) <= ...
                                this.d_boxes(ii)/2 + this.d_boxes(jj-1)/2 + 0.001 && ...
                                abs(this.x_boxes_vect(ii,this.cont-1) - this.x_boxes_vect(jj-1,this.cont-1)) >= ...
                                this.d_boxes(ii)/2 + this.d_boxes(jj-1)/2 && ...
                                abs(this.y_boxes(ii) - this.y_boxes(jj-1)) <= ...
                                this.d_boxes(ii)/2 + this.d_boxes(jj-1)/2 && ...
                                abs(this.y_boxes_vect(ii,this.cont-1) - this.y_boxes_vect(jj-1,this.cont-1)) <= ...
                                this.d_boxes(ii)/2 + this.d_boxes(jj-1)/2

                            if this.x_boxes(ii) - this.d_boxes(ii)/2 <= this.x_boxes(jj-1) + this.d_boxes(jj-1)/2 && ...
                                    this.x_boxes(ii) - this.d_boxes(ii)/2 > this.x_boxes(jj-1) - this.d_boxes(jj-1)/2

                                penetrazione_x = ...
                                    (this.x_boxes(jj-1) + this.d_boxes(jj-1)/2) - ...
                                    (this.x_boxes(ii) - this.d_boxes(ii)/2);

                                this.x_boxes(ii) = this.x_boxes(ii) + penetrazione_x/2 + this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) - penetrazione_x/2 - this.toll_contatto;

                            elseif this.x_boxes(ii) + this.d_boxes(ii)/2 >= this.x_boxes(jj-1) - this.d_boxes(jj-1)/2 && ...
                                    this.x_boxes(ii) + this.d_boxes(ii)/2 < this.x_boxes(jj-1) + this.d_boxes(jj-1)/2

                                penetrazione_x = ...
                                    (this.x_boxes(ii) + this.d_boxes(ii)/2) - ...
                                    (this.x_boxes(jj-1) - this.d_boxes(jj-1)/2);

                                this.x_boxes(ii) = this.x_boxes(ii) - penetrazione_x/2 - this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) + penetrazione_x/2 + this.toll_contatto;
                            end

                        elseif abs(this.y_boxes(ii) - this.y_boxes(jj-1)) <= ...
                                this.d_boxes(ii)/2 + this.d_boxes(jj-1)/2 + 0.001 && ...
                                abs(this.y_boxes_vect(ii,this.cont-1) - this.y_boxes_vect(jj-1,this.cont-1)) >= ...
                                this.d_boxes(ii)/2 + this.d_boxes(jj-1)/2 && ...
                                abs(this.x_boxes(ii) - this.x_boxes(jj-1)) <= ...
                                this.d_boxes(ii)/2 + this.d_boxes(jj-1)/2

                            if this.y_boxes(ii) - this.d_boxes(ii)/2 <= this.y_boxes(jj-1) + this.d_boxes(jj-1)/2 && ...
                                    this.y_boxes(ii) - this.d_boxes(ii)/2 > this.y_boxes(jj-1) - this.d_boxes(jj-1)/2

                                penetrazione_y = ...
                                    (this.y_boxes(jj-1) + this.d_boxes(jj-1)/2) - ...
                                    (this.y_boxes(ii) - this.d_boxes(ii)/2);

                                this.y_boxes(ii) = this.y_boxes(ii) + penetrazione_y/2 + this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) - penetrazione_y/2 - this.toll_contatto;

                            elseif this.y_boxes(ii) + this.d_boxes(ii)/2 < this.y_boxes(jj-1) + this.d_boxes(jj-1)/2 && ...
                                    this.y_boxes(ii) + this.d_boxes(ii)/2 >= this.y_boxes(jj-1) - this.d_boxes(jj-1)/2

                                penetrazione_y = ...
                                    (this.y_boxes(ii) + this.d_boxes(ii)/2) - ...
                                    (this.y_boxes(jj-1) - this.d_boxes(jj-1)/2);

                                this.y_boxes(ii) = this.y_boxes(ii) - penetrazione_y/2 - this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) + penetrazione_y/2 + this.toll_contatto;
                            end
                        end
                    end
                end

                if this.i_box(ii) < 6 && this.j_box(ii) < 6
                    act_AMS(this.j_box(ii) + (this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii) + (this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end

            %% Observation
            Observation = zeros(75,1);
            for ii = 1:this.n_boxes_tot
                if this.index_AMS(ii) > 0
                    idx = this.index_AMS(ii);
                    base = (idx-1)*3 + 1;
                    Observation(base)   = this.x_boxes(ii);
                    Observation(base+1) = this.y_boxes(ii);
                    Observation(base+2) = this.d_boxes(ii);
                end
            end

            %% Save trajectory history
            for ii = 1:this.max_gen_boxes
                if ii <= this.n_boxes_tot
                    this.x_boxes_vect(ii,this.cont) = this.x_boxes(ii);
                    this.y_boxes_vect(ii,this.cont) = this.y_boxes(ii);
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                else
                    this.x_boxes_vect(ii,this.cont) = -1;
                    this.y_boxes_vect(ii,this.cont) = -1;
                end
            end

            this.State = Observation;

            %% Terminal condition
            cont_exit = 0;

            for ii = 1:this.n_boxes_tot
                if this.y_boxes(ii) > this.d_AMS*5
                    cont_exit = cont_exit + 1;

                    if this.pack_exited(ii) == 0
                        this.pack_exited(ii) = 1;
                        this.index_exit = this.index_exit + 1;
                        this.exit_order(this.index_exit) = ii;
                    end
                end
            end

            IsDone = cont_exit == this.max_gen_boxes;
            this.IsDone = IsDone;

            %% Reward
            Reward = getReward(this, AMS_actions);

            notifyEnvUpdated(this);

        end

        function InitialObservation = reset(this)

            this.index_exit = 0;
            this.pack_exited = zeros(this.max_gen_boxes,1);
            this.exit_order = zeros(this.max_gen_boxes,1);

            this.cont_new_box = 1;

            this.n_boxes_tot_vect = 3;
            this.n_boxes_tot = this.n_boxes_tot_vect;

            this.time = 0;
            this.vy_boxes = zeros(3,1);
            this.cont = 0;

            max_d_box = 2.0 * this.d_AMS;
            min_d_box = 0.25 * this.d_AMS;

            l_AMS_matrix = this.d_AMS * this.n_j_AMS;

            %% Generate box sizes
            for ii = 1:this.max_gen_boxes
                this.d_boxes(ii) = min_d_box + (max_d_box - min_d_box)*rand(1);
            end

            %% Generate initial 3 boxes
            x1 = this.d_boxes(1)*0.5 + ...
                (this.d_AMS*2.5 - this.d_boxes(1)*0.5)*rand(1);

            x2 = this.d_AMS*2.5 + this.d_boxes(2)*0.5 + ...
                (l_AMS_matrix - (this.d_AMS*2.5 + this.d_boxes(2)*0.5))*rand(1);
            x3 = this.d_boxes(3)*0.5 + (l_AMS_matrix - this.d_boxes(3)*0.5)*rand(1);


            boxes_x = [x1, x2, x3];
            boxes_d = [this.d_boxes(1), this.d_boxes(2), this.d_boxes(3)];

            % Iteratively resolve overlaps (multiple passes in case fixes cause new overlaps)
            max_iter = 20;
            for iter = 1:max_iter
                any_overlap = false;

                for a = 1:3
                    for b = a+1:3
                        gap = abs(boxes_x(a) - boxes_x(b));
                        minDist = boxes_d(a)/2 + boxes_d(b)/2 + 0.1;

                        if gap < minDist
                            any_overlap = true;
                            correction = (minDist - gap) / 2;

                            if boxes_x(a) < boxes_x(b)
                                boxes_x(a) = boxes_x(a) - correction;
                                boxes_x(b) = boxes_x(b) + correction;
                            else
                                boxes_x(a) = boxes_x(a) + correction;
                                boxes_x(b) = boxes_x(b) - correction;
                            end
                        end
                    end

                    % Clamp each box to AMS bounds after every adjustment
                    half = boxes_d(a)/2;
                    if boxes_x(a) - half < 0
                        boxes_x(a) = half;
                    elseif boxes_x(a) + half > l_AMS_matrix
                        boxes_x(a) = l_AMS_matrix - half;
                    end
                end

                if ~any_overlap
                    break;
                end
            end

            x1 = boxes_x(1);
            x2 = boxes_x(2);
            x3 = boxes_x(3);




            y1 = 0.001 + rand(1)*0.01;
            y2 = 0.001 + rand(1)*0.05;
            y3 = 0.001 + rand(1)*0.05;



            this.x_boxes(1) = x1;
            this.y_boxes(1) = y1;

            this.x_boxes(2) = x2;
            this.y_boxes(2) = y2;


            this.x_boxes(3) = x3;
            this.y_boxes(3) = y3;

            this.x_boxes_prec(1) = x1;
            this.x_boxes_prec(2) = x2;

            this.y_boxes_prec(1) = y1;
            this.y_boxes_prec(2) = y2;

            this.x_boxes_prec(3) = x3;
            this.y_boxes_prec(3) = y3;

            for ii = this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = 0;
                this.y_boxes(ii) = 0;
            end

            this.index_AMS = zeros(3,1);

            %% Identify initial AMS cells
            for ii = 1:this.n_boxes_tot

                if this.y_boxes(ii) <= this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii) <= this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii) <= this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii) <= this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii) <= this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii) <= this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii) <= this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii) <= this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii) <= this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                if this.i_box(ii) < 6 && this.j_box(ii) < 6
                    this.index_AMS(ii) = this.j_box(ii) + (this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end

            %% Initial observation
            InitialObservation = zeros(75,1);
            for ii = 1:this.n_boxes_tot
                if this.index_AMS(ii) > 0
                    idx = this.index_AMS(ii);
                    base = (idx-1)*3 + 1;
                    InitialObservation(base)   = this.x_boxes(ii);
                    InitialObservation(base+1) = this.y_boxes(ii);
                    InitialObservation(base+2) = this.d_boxes(ii);
                end
            end
            this.State = InitialObservation;

            notifyEnvUpdated(this);

        end

    end

    %% Optional Methods
    methods

        function varargout = plot(this)

            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = AMSVisualizer_simple(this);
            else
                bringToFront(this.Visualizer);
            end

            if nargout
                varargout{1} = this.Visualizer;
            end

            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;

        end

        function Reward = getReward(this, AMS_actions) %#ok<INUSD>

            Reward = 0;

            % -----------------------------------------
            % AMS geometry
            % -----------------------------------------
            y_exit = this.d_AMS * this.n_i_AMS;
            l_AMS_matrix = this.d_AMS * this.n_j_AMS;
            x_center = l_AMS_matrix / 2;

            % -----------------------------------------
            % Reward parameters (original — unchanged)
            % -----------------------------------------
            safeGap = 0.02;
            gapPenaltyWeight = 2;
            gapGoodReward = 0.3;
            %rowSamePenalty = 0.15;
            safeMargin = 0.02;
            wallPenaltyWeight = 2;
            forwardRewardWeight = 0.05;

            centerRewardWeight = 0.03;
            overlap2D_maxPenalty    = 1;   % max penalty per axis (x or y) per pair
            %exitReward = 0.3;
            %exitSeparationBonus = 0.4;  % extra if exiting box is well-separated from next

            % -----------------------------------------
            % 1) Select parcels inside AMS only
            % -----------------------------------------
            eval_idx = [];
            for ii = 1:this.n_boxes_tot
                if this.y_boxes(ii) >= 0 && this.y_boxes(ii) <= y_exit
                    eval_idx(end+1) = ii; %#ok<AGROW>
                end
            end
            K = length(eval_idx);

            % -----------------------------------------
            % 2) Neighbor-based gap reward/penalty
            % -----------------------------------------
            if K >= 2
                y_positions = zeros(K, 1);
                for n = 1:K
                    y_positions(n) = this.y_boxes(eval_idx(n));
                end

                [y_sorted, ~] = sort(y_positions, 'descend');  % highest y = closest to exit

                % Reward the spread between leader and the pack
                % Large spread = good singulation
                leaderY  = y_sorted(1);
                secondY  = y_sorted(2);

                % Core singulation gap — leader vs second box
                singulationGap = leaderY - secondY;  % edge-to-edge would be cleaner but center is fine here

                if singulationGap >= safeGap
                    Reward = Reward + gapGoodReward * min(singulationGap / safeGap, 3.0);
                else
                    gapPenalty = gapPenaltyWeight * (safeGap - singulationGap) / safeGap;
                    Reward = Reward - min(gapPenalty, 3);
                end

                % Also penalize if 2nd and 3rd are clumped — encourage the whole queue to spread
                if K >= 3
                    thirdY = y_sorted(3);
                    secondGap = secondY - thirdY;

                    if secondGap >= safeGap
                        Reward = Reward + (gapGoodReward * 0.5) * min(secondGap / safeGap, 3.0);
                    else
                        gapPenalty = gapPenaltyWeight * (safeGap - secondGap) / safeGap;
                        Reward = Reward - min(gapPenalty * 0.5, 1.5);  % half weight vs leader gap
                    end
                end
            end

            % -----------------------------------------
            % 3) Smooth wall penalty
            % -----------------------------------------
            wallPenalty = 0;
            for n = 1:K
                ii = eval_idx(n);
                clearanceLeft  = this.x_boxes(ii) - this.d_boxes(ii)/2;
                clearanceRight = l_AMS_matrix - (this.x_boxes(ii) + this.d_boxes(ii)/2);
                clearance = min(clearanceLeft, clearanceRight);
                if clearance < safeMargin
                    wallPenalty = wallPenalty + (safeMargin - clearance) / safeMargin;
                end
            end
            Reward = Reward - wallPenaltyWeight * wallPenalty;

            % -----------------------------------------
            % 4) Small forward motion reward
            % -----------------------------------------
            vy_max_achievable = this.upperlim_v * cos(abs(this.lowerlim_rot)); % ~1.556

            for n = 1:K
                ii = eval_idx(n);

                if this.y_boxes_prec(ii) <= 0
                    continue  % skip newly spawned boxes
                end

                vy_actual = (this.y_boxes(ii) - this.y_boxes_prec(ii)) / this.dt;

                if vy_actual > 0
                    vy_norm = min(vy_actual / vy_max_achievable, 1.0);
                    Reward = Reward + forwardRewardWeight * vy_norm;
                end
            end

            % -----------------------------------------
            % 5) Very small center-position reward
            % -----------------------------------------
            for n = 1:K
                ii = eval_idx(n);
                dist_from_center = abs(this.x_boxes(ii) - x_center);
                centerScore = 1 - dist_from_center / x_center;
                centerScore = max(0, min(centerScore, 1));
                Reward = Reward + centerRewardWeight * centerScore;
            end

            % -----------------------------------------
            % 6) Exit reward — bonus for separation at exit
            % -----------------------------------------

            %
            % for ii = 1:this.n_boxes_tot
            %
            %     justExited = this.pack_exited(ii) == 0 && this.y_boxes(ii) > y_exit;
            %
            %     if justExited
            %         Reward = Reward + exitReward;
            %
            %         % Find the closest still-on-AMS box by y position
            %         minDist = inf;
            %         for jj = 1:this.n_boxes_tot
            %             if jj ~= ii && this.y_boxes(jj) <= y_exit && this.y_boxes(jj) > 0
            %                 dist = abs(this.y_boxes(ii) - this.y_boxes(jj));
            %                 if dist < minDist
            %                     minDist = dist;
            %                 end
            %             end
            %         end
            %
            %         % Bonus scaled by how separated it is from the next box
            %         if minDist > safeGap
            %             sepScore = min((minDist - safeGap) / safeGap, 1.0);
            %             Reward = Reward + exitSeparationBonus * sepScore;
            %         end
            %     end
            % end
            %


            % =========================================
            % 7) 2D edge-to-edge safe-gap penalty
            %    Boxes are penalized only if they are
            %    closer than safeGap in BOTH x and y.
            % =========================================
            if K >= 2

                for a = 1:K-1
                    for b = a+1:K

                        i1 = eval_idx(a);
                        i2 = eval_idx(b);

                        % -----------------------------
                        % Box 1 edges from center
                        % -----------------------------
                        xLeft1  = this.x_boxes(i1) - this.d_boxes(i1)/2;
                        xRight1 = this.x_boxes(i1) + this.d_boxes(i1)/2;

                        yBot1 = this.y_boxes(i1) - this.d_boxes(i1)/2;
                        yTop1 = this.y_boxes(i1) + this.d_boxes(i1)/2;

                        % -----------------------------
                        % Box 2 edges from center
                        % -----------------------------
                        xLeft2  = this.x_boxes(i2) - this.d_boxes(i2)/2;
                        xRight2 = this.x_boxes(i2) + this.d_boxes(i2)/2;

                        yBot2 = this.y_boxes(i2) - this.d_boxes(i2)/2;
                        yTop2 = this.y_boxes(i2) + this.d_boxes(i2)/2;

                        % -----------------------------
                        % Edge-to-edge gaps
                        % gap > 0  : separated
                        % gap = 0  : touching
                        % gap < 0  : overlapping
                        % -----------------------------
                        gapX = max(xLeft1, xLeft2) - min(xRight1, xRight2);
                        gapY = max(yBot1, yBot2) - min(yTop1, yTop2);

                        % -----------------------------
                        % Penalize only if safe gap is violated
                        % in BOTH directions
                        % -----------------------------
                        if gapX < safeGap && gapY < safeGap

                            violationX = safeGap - gapX;
                            violationY = safeGap - gapY;

                            % Normalize using smaller box size
                            minSize = min(this.d_boxes(i1), this.d_boxes(i2));

                            penX = (violationX / minSize) * overlap2D_maxPenalty;
                            penY = (violationY / minSize) * overlap2D_maxPenalty;

                            % Cap each penalty
                            penX = min(penX, overlap2D_maxPenalty);
                            penY = min(penY, overlap2D_maxPenalty);

                            Reward = Reward - (penX + penY);

                        end

                    end
                end

            end

        end

    end


    %% Protected callback
    methods (Access = protected)

        function envUpdatedCallback(this)
        end

    end

end
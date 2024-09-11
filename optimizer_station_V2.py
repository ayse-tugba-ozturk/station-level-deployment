import math
import timeit
import warnings
import cvxpy as cp
print("CVXPY version:", cp.__version__)
import numpy as np
import copy

from scipy.special import softmax

warnings.filterwarnings("ignore")

class Parameters:
    """
    Class to hold all parameters for all simulations which will be the same for each problem.
    """
    def __init__(self, 
                z0 = np.array([1,1,1,1]).reshape(4,1),
                v0 = np.array([0.3333, 0.3333, 0.3333]).reshape(3,1) ,  
                Ts = 0.25, # 1 control horizon interval = 0.25 global hour
                base_tarriff_overstay = 1.0, 
                eff = 1,  # Charging efficient, assumed to be 100%
                soft_v_eta = 1e-4, # For convex relaxation in constraints.
                opt_eps = 0.0001, 
                TOU = np.ones((96,)),
                demand_charge_cost = 18): # Time-of-user charging tariff: 96 intervals, 1 day.

        # TOU Tariff in cents / kwh
        # TOU * power rate = cents  / hour

        self.v0 = v0
        self.z0 = z0
        # print("z0:",z0)
        self.Ts = Ts
        self.base_tariff_overstay = base_tarriff_overstay
        self.TOU = TOU
        self.eff = eff # power efficiency
        self.dcm_choices = ['charging with flexibility', 'charging asap', 'leaving without charging']
        self.soft_v_eta = soft_v_eta #softening equality constraint for v; to avoid numerical error
        self.opt_eps = opt_eps
        self.cost_dc = demand_charge_cost  # Cost for demand charge. This value is arbitrary now. A larger value means the charging profile will go average.
        # 18.8 --> 300 We can change this value to show the effect of station-level impact.

        assert len(self.TOU) == int(24 / self.Ts), "Mismatch between TOU cost array size and discretization steps"

class Problem:
    """
    This class encompasses the current user information which will change for every user optimization.

    time, int, user interval
    duration, int, number of charging intervals
    """
    def __init__(self, par,station_info, k, **kwargs):
        self.Parameters = par
        self.station_info = copy.deepcopy(station_info) if station_info else None # "station" ndarray.
        event = kwargs["event"]
        self.event = event
        self.k = k 
        self.user_time = event["time"]
        self.e_need = event["e_need"]
        self.historical_peak = event["historical_peak"]

        self.user_duration = event["duration"]
        # self.user_overstay_duration = round(event["overstay_duration"] / par.Ts) * par.Ts
        # Power cap for the station charger
        self.station_pow_max = event["station_pow_max"]
        self.user_power_rate = event['user_power_rate']

        self.power_rate = min(self.user_power_rate, self.station_pow_max) # The actual power cap for user

        self.dcm_charging_sch_params = np.array([[ - self.power_rate * 0.0184 / 2], [self.power_rate * 0.0184 / 2], [0], [0]])
        #% DCM parameters for choice 1 -- charging with flexibility
        self.dcm_charging_reg_params = np.array([[self.power_rate * 0.0184 / 2], [- self.power_rate * 0.0184 / 2], [0], [0.341]])
        #% DCM parameters for choice 2 -- charging as soon as possible
        self.dcm_leaving_params = np.array([[self.power_rate * 0.005 / 2], [self.power_rate * 0.005 / 2], [0], [-1]])
        
        #% DCM parameters for choice 3 -- leaving without charging
        self.THETA = np.vstack((self.dcm_charging_sch_params.T, self.dcm_charging_reg_params.T, self.dcm_leaving_params.T))

        # problem specifications
        self.N_sch = self.user_duration
        self.N_reg = math.ceil((self.e_need / self.power_rate / par.eff * int(1 / par.Ts)))
        self.N_reg_remainder = (self.e_need / self.power_rate / par.eff * int(1 / par.Ts)) % 1

        self.assertion_flag = 0
        ## Option 1: Update the e_need to avoid opt failure
        if self.N_sch < self.N_reg:
            # self.e_need = self.N_sch * self.power_rate * par.eff * par.Ts
            # self.assertion_flag = 1
        ## Option 2: Update the N_sch to avoid opt failure
            self.N_sch = self.N_reg
            self.user_duration = self.N_sch
            self.assertion_flag = 1

        if len(par.TOU) < self.user_time + self.user_duration: # if there is overnight charging
            par.TOU = np.concatenate([par.TOU,par.TOU])
        self.TOU = par.TOU[self.user_time:(self.user_time + self.user_duration)]

        self.limit_reg_with_sch = event["limit_reg_with_sch"]
        self.limit_sch_with_constant = event["limit_sch_with_constant"]
        self.sch_limit = event["sch_limit"] if self.limit_sch_with_constant else None

        # ###### CREATE THE STATION INFO 
        # self.k =event["time"] # Current global time indices(hour unit, for example 1.0, 1.25, 1.5, 1.75, 2.0...)
    

        ### Existing FLEX Users
        station = dict()
        self.station = station
        station["REG_list"] = list()
        station["SCH_list"] = list()
        if self.station_info:
            station_info = self.station_info # External input: all information about the states
            for user in station_info:
                if user["choice"] == "REG":
                    station["REG_list"].append(user["dcosId"])
                elif user["choice"] == "SCH":
                    station["SCH_list"].append(user["dcosId"])


        # Update existing user info(e_needed), transform the input dict to internal nparray
        user_keys = station['SCH_list']
        num_sch_user = len(station['SCH_list'])
        if user_keys:
            users = [d for d in station_info if d["dcosId"] in user_keys]
            for i in range(len(user_keys)):
                user = users[i]
                user["start_time"] = int(user["start_time"] / self.Parameters.Ts) # The time user arrives
                user["end_time"] = int(user["end_time"] / self.Parameters.Ts) # The time user leaves
                # Number of intervals left for the existing users
                user["N_remain"] = int(user["end_time"] - self.k / self.Parameters.Ts)
                # Current local time indices for User i
                user["TOU_idx"] = int(self.k / self.Parameters.Ts - user["start_time"])
                user["TOU"] = self.Parameters.TOU[user["start_time"]: user["end_time"]].reshape(-1, 1)
                # How much power we already charged?
                user["energyNeeded"] = user["energyNeeded"] - np.sum(user["optPower"][: user["TOU_idx"]] * self.Parameters.eff * self.Parameters.Ts)


        # For the REG users, integrate their information in reg_user_info list.
        user_keys = station["REG_list"]
        if user_keys:
            users = [d for d in station_info if d["dcosId"] in user_keys]
            for i in range(len(user_keys)):
                user = users[i]
                user["start_time"] = int(user["start_time"] / self.Parameters.Ts)
                user["end_time"] = int(user["end_time"] / self.Parameters.Ts)
                user["N_remain"] = int(user["end_time"] - self.k / self.Parameters.Ts) # Number of intervals left
                user["TOU_idx"] = int(self.k / self.Parameters.Ts - user["start_time"])
                user["TOU"] = self.Parameters.TOU[user["start_time"]: user["end_time"]].reshape(-1, 1)
                user["N_reg"] = math.ceil((user["energyNeeded"] / user["power_rate"] / self.Parameters.eff * int(1 / self.Parameters.Ts)))

        ### New User information & All SCH users info (existing_user_info)
        if self.station_info:
            N_remain_all = [user["N_remain"] for user in station_info]
        else:
            N_remain_all = []
        N_remain_all.append(self.N_sch)

        var_dim_constant = int(max(N_remain_all)) # maximum remaining duration of all users


        self.var_dim_constant = var_dim_constant
        self.num_sch_user = num_sch_user
        self.station = copy.deepcopy(station) if station else None
        self.k = k # Current global time indices(hour unit, for example 1.0, 1.25, 1.5, 1.75, 2.0...)
class Optimization_station:
    """
    This class encompasses the main optimizer at the station level.
    """
    def __init__(self, par, prb, k):
        self.Parameters = par
        self.Problem = prb
        self.station = prb.station
        self.opt_z = None
        self.opt_tariff_asap = None
        self.opt_tariff_flex = None
        self.opt_tariff_overstay = None
        self.station_info = prb.station_info
        self.k = k # Current global time indices(hour unit, for example 1.0, 1.25, 1.5, 1.75, 2.0...)
        # maximum remaining duration of all users
        self.var_dim_constant = prb.var_dim_constant
        
    def argmin_v(self, u, z, p_dc_sch, p_dc_reg):
        """
        Parameters 
        Decision Variables: 
        v: softmax probability[ sm(theta_sch, z), sm(theta_reg, z), sm(theta_leave, z) ], shape: (3,1)
        """
        ### Read parameters
        THETA = self.Problem.THETA 
        soft_v_eta = self.Parameters.soft_v_eta

        ### Decision Variables
        v = cp.Variable(shape = (3), pos = True)

        ### Objective Function
        J, J_array,_,_ = self.constr_J(u, z, v, p_dc_sch, p_dc_reg)
        ### Log sum function conjugate: negative entropy 
        # lse_conj = - cp.sum(cp.entr(v))
        # func = v.T @ (THETA @ z)
        # # J_4 = mu * (lse_conj - func) 
        # constraints += [ v <= np.array((1,1,1))] # What is this? 
        # constraints += [ cp.sum(v) >= 1 - soft_v_eta ]

        ### Constraints
        constraints = [v >= 0]
        constraints += [cp.sum(v) == 1]
        constraints += [cp.log_sum_exp(THETA @ z) - cp.sum(cp.entr(v)) - v.T @ (THETA @ z) <= soft_v_eta]
        
        ### Solve
        obj = cp.Minimize(J - (cp.sum(cp.entr(v)) + v.T @ (THETA @ z)))
        prob = cp.Problem(obj, constraints)
        try: 
            prob.solve(solver="GUROBI",verbose=False)
        except Exception as e: 
            # print(e)
            try:
                # print("GURUOBI failed, use ECOS")
                prob.solve(solver="ECOS")
            except Exception as e: 
                # print(e)
                try:
                    prob.solve(solver="ECOS")
                except: 
                    # print(softmax(THETA @ z))
                    bruteforce=softmax(THETA @ z).reshape(3,1)
                    return bruteforce


        return np.round(v.value, 4)
    def argmin_z(self, u, v, p_dc_sch, p_dc_reg):
        """
        Function to determine prices 

        Decision Variables: 
        z: price [tariff_flex, tariff_asap, tariff_overstay, leave = 1 ]

        Parameters: 
        u, array, power for flex charging 
        v, array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
        lam_x, regularization parameter for sum squares of the power var (u)
        lam_z_c, regularization parameter for sum squares of the price flex (u)
        lam_z_uc, regularization parameter for sum squares of the price asap (u)
        lam_h_c, regularization parameter for g_flex
        lam_h_uc, regularization parameter for g_asap
        N_sch: timesteps arrival to departure 
        N_reg: timesteps required when charging at full capacity

        """
        # if sum(v) < 0 | (np.sum(v) < 1 - self.Parameters.soft_v_eta) | (np.sum(v) > 1 + self.Parameters.soft_v_eta):
        #     raise ValueError('[ ERROR] invalid $v$')
        
        ### Read parameters
        soft_v_eta = self.Parameters.soft_v_eta
        THETA = self.Problem.THETA

        ### Decision Variables
        z = cp.Variable(shape = (4), pos = True)

        ### Objective Function
        J, _ , _,_= self.constr_J(u, z, v, p_dc_sch, p_dc_reg)

        ### Constraints
        constraints = [z[3] == 1]
        constraints = [z[2] == 1]
        constraints += [cp.log_sum_exp(THETA @ z) - cp.sum(cp.entr(v)) - v.T @ (THETA @ z) <= soft_v_eta]
        constraints += [z <= 45] # For better convergence guarantee.

        if self.Problem.limit_reg_with_sch:
            constraints += [z[1] <= z[0]]
        if self.Problem.limit_sch_with_constant:
            constraints += [z[0] == self.Problem.sch_limit]
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver='GUROBI',verbose=False)
        except Exception as e:
            # print(e)
            prob.solve(verbose=False)

        return z.value
    def argmin_u(self, z, v):
        """
        Function to minimize charging cost. Flexible charging with variable power schedule
        Inputs: 

        Parameters: 
        z, array where [tariff_flex, tariff_asap, tariff_overstay, leave = 1 ]
        v, array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
        lam_x, regularization parameter for sum squares of the power var (u)
        lam_h_c, regularization parameter for g_flex
        lam_h_uc, regularization parameter for g_asap
        N_sch: timesteps arrival to departure 
        N_reg: timesteps required when charging at full capacity

        Parameters: 
        Decision Variables:
        SOC: state of charge (%)
        u: power (kW)

        Objective Function:
        Note: delta_k is not in the objective function!! 
        Check if it could make difference since power is kW cost is per kWh 

        Outputs
        u: power 
        SOC: SOC level 
        """

        ### Read parameters

        eff = 1
        delta_t = self.Parameters.Ts
        historical_peak = self.Problem.historical_peak

        ### Decision Variables
        num_sch_user = len(self.station["SCH_list"]) + 1 # num of all SCH users
        e_delivered = cp.Variable(shape = ((self.var_dim_constant + 1) * num_sch_user, 1))
        u = cp.Variable(shape = (self.var_dim_constant * num_sch_user, 1))
        p_dc_sch = cp.Variable(shape = 1)
        p_dc_reg = cp.Variable(shape = 1)
        # p_demand_charge_sch = cp.Variable(shape = (1, 1))
        # p_demand_charge_reg = cp.Variable(shape = (1, 1))

        ### Objective Function
        

        ### Constraints (should incorporate all SCH users)
        constraints = [u >= 0]
        # The following constraints iterates through all existing flex users
        for i in range(num_sch_user):  # For all possible SCH users
            if i == 0:  # For the new user
                N_remain = self.Problem.N_sch
                e_need = self.Problem.e_need
                power_rate = self.Problem.power_rate

            else: # For existing users
                user_key = self.station["SCH_list"][i - 1]
                user = [d for d in self.station_info if d["dcosId"] == user_key][0]
                N_remain = int(user["N_remain"])
                e_need = float(user["energyNeeded"])
                power_rate = float(user["power_rate"])

            # Shape of e_delivered: (num_sch * (self.var_dim_constant + 1), 1)
            # Shape of u: (num_sch * self.var_dim_constant, 1)
            if e_need < 0:
                print("E_need: {}, Energy demand should be greater than 0".format(e_need))
                e_need = 0 
            N_need = (e_need / power_rate) * 4

            if N_need >= N_remain:
                # print("N_need:{} N_remain {}".format(N_need,N_remain))
                e_need = (N_remain * power_rate / 4) - 0.01
           
            
                


            e_start = int(i * (self.var_dim_constant + 1))
            e_end = int(i * (self.var_dim_constant + 1) + N_remain)
            e_max = int(i * (self.var_dim_constant + 1) + self.var_dim_constant)
            u_start = int(i * self.var_dim_constant)
            u_end = int(i * self.var_dim_constant + N_remain)

            constraints += [u[u_start: u_end] <= power_rate]
            constraints += [e_delivered[e_start] == 0]
            constraints += [e_delivered[e_end] >= e_need]
            constraints += [e_delivered[e_start: e_max+1] <= e_need]


            # Implication: e_end = e_need.

            # Charging dynamics within each user
            for j in range(self.var_dim_constant):
                constraints += [e_delivered[j + e_start + 1] == e_delivered[j + e_start] + (float(eff) * delta_t * u[u_start + j])]

        ## Solve 
        J, _, current_peak_sch, current_peak_reg= self.constr_J(u, z, v, p_dc_sch, p_dc_reg)

        # Demand charge constraints
        constraints += [historical_peak <= p_dc_sch]
        constraints += [historical_peak <= p_dc_reg]
        constraints += [current_peak_sch <= p_dc_sch]
        constraints += [current_peak_reg <= p_dc_reg]


        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        if prob.status != 'optimal':
            print(prob.status)
            print("Gurobi failed, cant solve for power")
            prob.solve(solver='GUROBI',verbose=False)
            # print(e_need)

        return u.value, e_delivered.value, p_dc_sch.value, p_dc_reg.value

    def constr_J(self, u, z, v, p_dc_sch, p_dc_reg):

        ### Read parameters for the new session
        N_reg = self.Problem.N_reg
        N_sch = self.Problem.N_sch
        TOU = self.Problem.TOU
        power_rate = self.Problem.power_rate
        N_reg_remainder = self.Problem.N_reg_remainder
        delta_t = self.Parameters.Ts
        historical_peak  = self.Problem.historical_peak 
        num_sch_user = self.Problem.num_sch_user

        ### Retrieve parameters for existing users
        existing_sch_obj = 0
        user_keys = self.station['SCH_list']
        num_sch_user = len(user_keys) + 1
        if user_keys:
            users = [d for d in self.station_info if d["dcosId"] in user_keys]
            for i in range(1, num_sch_user):  # EVs other than the new user
                # Here we need "i - 1", since the first row of existing_user_info is a new user
                user = users[i - 1]
                adj_constant = int(i * self.var_dim_constant)
                N_remain = user["N_remain"]
                TOU_idx = user["TOU_idx"]

                existing_sch_obj += u[adj_constant: (adj_constant + N_remain)].T @ (user["TOU"][TOU_idx:] - user["price"]).reshape(-1, 1)

        existing_reg_obj = 0
        user_keys = self.station['REG_list']
        if user_keys:
            users = [d for d in self.station_info if d["dcosId"] in user_keys]
            TOU_idx = np.int_(self.k / delta_t - np.array([user["start_time"] for user in users]))
            existing_reg_obj = np.sum([user["optPower"][TOU_idx[i]: user["N_reg"]].T @ (
                    user["TOU"][TOU_idx[i]: user["N_reg"]] - user["price"]).reshape(-1, 1) for i, user
                                       in enumerate(users)])

        ## Existing user charging profile summation
        # REG
        reg_power_sum_profile = np.zeros(self.var_dim_constant)
        if self.station['REG_list']:
            users = [d for d in self.station_info if d["dcosId"] in self.station['REG_list']]
            for i in range(len(self.station['REG_list'])): # for all ASAP users
                user = users[i]
                TOU_idx = int(self.k / delta_t - user["start_time"])
                reg_power_sum_profile[: user["N_reg"] - TOU_idx] += user["optPower"][TOU_idx:].reshape(-1)
                # N_remain = N_reg - TOU_idx

        # SCH
        num_sch = len(self.station["SCH_list"]) + 1
        # Row: # of user, Col: Charging Profile
        sch_power_sum_profile = cp.reshape(u, (self.var_dim_constant, num_sch)).T
        sch_power_sum_profile = cp.sum(sch_power_sum_profile[1:, :], axis=0) # Shape: (self.var_dim_constant,)
        # The shape of sch_power_sum_profile is (self.var_dim_constant, 1)

        ## New user charging profile(ASAP)
        reg_new_user_profile = np.zeros(self.var_dim_constant)
        reg_new_user_profile[: N_reg - 1] = power_rate
        reg_new_user_profile[N_reg - 1] = (power_rate * N_reg_remainder) if N_reg_remainder > 0 else power_rate

        c_co = cp.reshape((TOU[:N_sch] - z[0]), (N_sch, 1))

        # Use cvxpy so all variables are cvxpy variables
        new_sch_obj = (u[: N_sch].T @ c_co) * delta_t
        new_reg_obj = (cp.sum(power_rate * (TOU[:N_reg - 1] - z[1])) + (power_rate * N_reg_remainder) * (
                    TOU[N_reg - 1] - z[1])) * delta_t if N_reg_remainder > 0 else cp.sum(
            power_rate * (TOU[:N_reg] - z[1])) * delta_t
        new_leave_obj = 0

        new_sch_obj = new_sch_obj.flatten()[0]
        new_reg_obj = new_reg_obj.flatten()[0]
    
        # We dont actually need these line here 
        current_peak_sch = cp.max(reg_power_sum_profile + cp.sum(cp.reshape(u, (self.var_dim_constant, num_sch)).T, axis=0))
        current_peak_reg =  cp.max(reg_power_sum_profile + sch_power_sum_profile + reg_new_user_profile)
        # print("reg_peak: ",current_peak_reg.shape )

        # # first row of the u array is the 
        J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * (p_dc_sch-historical_peak))* v[0]
        J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * (p_dc_reg-historical_peak)) * v[1]
        J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj + self.Parameters.cost_dc * 0) * v[2]

        J = J0 + J1 + J2
        print()

        try:
            J_array = np.array([J0.value, J1.value, J2.value])
        # Because there is no vehicle do optimize when LEAVE this is not an opt. variable its a deterministic value
        except: 
            J_array = np.array([J0.value, J1.value, J2])

        return J, J_array, current_peak_sch, current_peak_reg


    def run_opt(self, solution_algorithm):
        """ BCD or grid_search"""
        ### This is the main optimization function (multi-convex principal problem)
        ### station_info: a copy of the state dictionary. The re-optimized profiles will be passed in this dictionary.
        var_dim_constant = self.var_dim_constant
        num_sch_user = self.Problem.num_sch_user
        station = self.station

        start = timeit.timeit()
        # Initial values for decision variable U: Shape: (all possible SCH users * dimension_constant, 1)
        uk_flex = self.Problem.power_rate * np.zeros([var_dim_constant * (num_sch_user + 1), 1])

        p_dc_sch_k = self.Problem.historical_peak
        p_dc_reg_k = self.Problem.historical_peak

        if solution_algorithm == "BCD":
            itermax = 1000
            count = 0
            improve = np.inf

            zk = self.Parameters.z0
            # print("zk:",zk)
            vk = self.Parameters.v0

            ### THIS VALUES ARE STORED FOR DEBUGGING
            Jk = np.zeros((itermax))
            rev_flex = np.zeros((itermax))
            rev_asap = np.zeros((itermax))
            z_iter = np.zeros((4,itermax))
            v_iter = np.zeros((3,itermax))
            J_sub = np.zeros((3,itermax))

            while (count < itermax) & (improve >= 0) & (abs(improve) >= 0.00001):

                _, J_array,_,_ = self.constr_J(uk_flex, zk, vk, p_dc_sch_k,p_dc_reg_k)

                Jk[count] = J_array.sum()
                J_sub[:, count] = J_array.reshape(3,)
                # rev_flex[count], rev_asap[count] = charging_revenue(zk, uk_flex)
                z_iter[:, count] = zk.reshape((4,))
                v_iter[:, count] = vk.reshape((3,))

                # try:
                uk_flex, e_deliveredk_flex, p_dc_sch_k, p_dc_reg_k = self.argmin_u(zk, vk)
                # except:
                    # print('uk is not updated')
                    # pass


                vk = self.argmin_v(uk_flex, zk, p_dc_sch_k, p_dc_reg_k)

                try:
                    zk = self.argmin_z(uk_flex, vk, p_dc_sch_k, p_dc_reg_k)
                except Exception as e:
                    print(e)
                    print("zk is not updated")
                    pass

                _, J_array,_,_= self.constr_J(uk_flex, zk, vk, p_dc_sch_k, p_dc_reg_k)
                improve = Jk[count] - J_array.sum()
                print(improve)
                # print(J_func(zk, uk_flex, vk))
                count += 1

                if count >= 50:
                    print("Too much time for iteration(iteration times exceed 50)")
                    break
        elif solution_algorithm == "grid_search":

            
                        # Define the values for z_reg and z_sch
            values = np.array([20, 25, 30, 35, 40, 45])
            # Create combinations where z_reg >= z_sch
            tariff_grid= [(z_sch, z_reg) for z_reg in values for z_sch in values if z_reg >= z_sch]
            # Convert the list of tuples into a numpy array
            
            zk = self.Parameters.z0
            vk = self.Parameters.v0
            zk[3] = 1
            zk[2] = 1
            J_opt = 10000
            z_opt = zk
            uk_flex, e_deliveredk_flex, p_dc_sch_k, p_dc_reg_k = self.argmin_u(zk, vk)
            print("DC_SCH:", p_dc_sch_k, "DC_REG:", p_dc_reg_k)
            grid_search_results = {}

            for (z_sch_k, z_reg_k) in tariff_grid:
                
                zk[0] =  z_sch_k
                zk[1] =  z_reg_k
                vk = softmax(self.Problem.THETA @ zk).reshape(3,1)
                J, J_array, _, _ = self.constr_J(uk_flex, zk, vk, p_dc_sch_k, p_dc_reg_k)
                grid_search_results[ (z_sch_k, z_reg_k) ] = {}
                grid_search_results[ (z_sch_k, z_reg_k) ]["J"] =  J_array.sum()
                grid_search_results[ (z_sch_k, z_reg_k) ]["J_arr"] =  J_array
                # if J.value < J_opt:
                #     z_opt = zk
                #     J_opt = J.value
                #     J_opt_array = J_array
                #     print(z_opt[0] * 6.6 ,z_opt[1] * 6.6, J_opt)
            
            # count = None
        else:
            print("Error in solution algorithm")

        user_keys = station["SCH_list"]
        if user_keys:
            users = [d for d in station_info if d["dcosId"] in user_keys]
            for i in range(len(user_keys)):
                user = users[i] # SCH User i
                N_remain = user["N_remain"]
                TOU_idx = user["TOU_idx"]
                # Update the power profile, shape: (-1, 1)
                user["optPower"][TOU_idx:] = uk_flex[int((i + 1) * var_dim_constant): int((i + 1) * var_dim_constant + N_remain)].reshape(-1)

        ### Track the maximum power
        # REG
        reg_power_sum_profile = np.zeros(self.var_dim_constant)
        if self.station['REG_list']:
            users = [d for d in self.station_info if d["dcosId"] in self.station['REG_list']]
            for i in range(len(self.station['REG_list'])): # for all ASAP users
                user = users[i]
                TOU_idx = int(self.k / self.Parameters.Ts - user["start_time"])
                reg_power_sum_profile[: user["N_reg"] - TOU_idx] += user["optPower"][TOU_idx:].reshape(-1)
        # SCH
        num_sch = len(self.station["SCH_list"]) + 1
        # Row: # of user, Col: Charging Profile
        sch_power_sum_profile = uk_flex.reshape(num_sch, self.var_dim_constant)
        sch_power_sum_profile = np.sum(sch_power_sum_profile[1:, :], axis=0) # Shape: (self.var_dim_constant,)
        # The shape of sch_power_sum_profile is (self.var_dim_constant, 1)

        ## New user charging profile(ASAP)
        reg_new_user_profile = np.zeros(self.var_dim_constant)
        reg_new_user_profile[: self.Problem.N_reg - 1] = self.Problem.power_rate
        reg_new_user_profile[self.Problem.N_reg - 1] = (self.Problem.power_rate * self.Problem.N_reg_remainder) if self.Problem.N_reg_remainder > 0 else self.Problem.power_rate

        sch_agg = reg_power_sum_profile + np.sum(uk_flex.reshape(num_sch, self.var_dim_constant), axis=0)
        reg_agg = reg_power_sum_profile + sch_power_sum_profile + reg_new_user_profile
        sch_max = np.max(sch_agg)
        reg_max = np.max(reg_agg)

        # Calculate the utility cost: TOU * power + DC_cost * peak_power(cents/kWh * kW * intervals)

        TOU_all = self.Parameters.TOU
        if len(self.Parameters.TOU) < self.Problem.user_time + self.var_dim_constant: # if there is overnight charging
            TOU_all = np.concatenate([self.Parameters.TOU, self.Parameters.TOU])
        TOU_all = TOU_all[self.Problem.user_time:(self.Problem.user_time + self.var_dim_constant)].reshape(-1, 1)
        utility_cost_sch = np.sum(TOU_all.T @ sch_agg.reshape(-1, 1)) + self.Parameters.cost_dc * p_dc_sch_k[0]
        utility_cost_reg = np.sum(TOU_all.T @ reg_agg.reshape(-1, 1)) + self.Parameters.cost_dc * p_dc_reg_k[0]

        ### Output the results
        opt = dict()
        opt['e_need'] = self.Problem.e_need

        # Part 1: Prices & Utility Cost
        opt["z"] = zk
        opt["z_hr"] = zk * self.Problem.power_rate
        # cents / kwh and cents / hour
        opt["tariff_sch"] = zk[0]
        opt["tariff_reg"] = zk[1]
        opt["sch_centsPerHr"] = opt["z_hr"][0]
        opt["reg_centsPerHr"] = opt["z_hr"][1]

        opt["utility_cost_sch"] = utility_cost_sch
        opt["utility_cost_reg"] = utility_cost_reg
        opt['TOU'] = self.Parameters.TOU

        # Part 2: Power Profiles
        opt["power_rate"] = self.Problem.power_rate

        ### 'new_peak_sch': max(sch_max, historical_peak)

        opt['new_peak_sch'] = p_dc_sch_k
        opt['new_peak_reg'] = p_dc_reg_k

        opt["sch_e_delivered"] = e_deliveredk_flex
        N_remain = int(self.Problem.user_duration)
        opt["sch_powers"] = uk_flex[: N_remain]

        # For a possible "NEW" "ASAP" user, we assume that it's at the maximum for all ASAP intervals
        reg_powers = np.ones((self.Problem.N_reg, 1)) * self.Problem.power_rate
        if self.Problem.N_reg_remainder != 0: # For the last time slot, ASAP may not occupy the whole slot.
            reg_powers[self.Problem.N_reg - 1] = self.Problem.power_rate * self.Problem.N_reg_remainder
        opt["reg_powers"] = reg_powers

        # If the REG and SCH are actually identical, we pick REG as final value.

        if self.Problem.assertion_flag == 1:
            opt["sch_powers"] = reg_powers

        opt["sch_max"] = sch_max
        opt["reg_max"] = reg_max
        opt["sch_agg"] = sch_agg
        opt["reg_agg"] = reg_agg

        # Part 3: Probability & Iteration Parameters Output

        opt["v"] = vk
        opt["prob_flex"] = vk[0]
        opt["prob_asap"] = vk[1]
        opt["prob_leave"] = vk[2]

        opt["N_sch"] = self.Problem.N_sch
        opt["N_reg"] = self.Problem.N_reg

        try:
            
            opt["J_array"] = J_array
        except:
            opt['J'] = J_opt

        try:
            opt['grid_search_results'] = grid_search_results
        except:
            opt['grid_search_results'] = None


        # opt["num_iter"] = count
        opt["time_start"] = self.Problem.user_time
        opt["time_end_SCH"] = self.Problem.user_time + self.Problem.user_duration
        opt["time_end_REG"] = self.Problem.user_time + self.Problem.N_reg
        opt["time_max"] = self.Problem.user_time + self.var_dim_constant


        # Part 4: General Problem Space
        opt["prb"] = self.Problem
        opt["par"] = self.Parameters

        end = timeit.timeit()

        station_info = self.Problem.station_info
        

        return station_info, opt
    
# def grid_search():
#     ## Define prices to be searched

#     ## Find the optimal power vector

#     uk_flex, e_deliveredk_flex, p_dc_sch_k, p_dc_reg_k = self.argmin_u(zk, vk)

def main():
    ### Test ###
    from deployment_utils import  emptyStateRecord,currentTime15min,TOU_15min,data_format_convertion, unixTime
    import pandas as pd
    optTime = currentTime15min() ## currentTime rounded down to 15min 
    optHour=optTime.hour
    optYear = optTime.year
    optMonth = optTime.month
    optDay= optTime.day
    
    
    newOptDate = pd.Timestamp(year = optYear, 
                              month = optMonth, 
                              day = optDay, 
                              hour = 0
                              )
    ##State record 

    emptyState = emptyStateRecord(newOptDate)

    emptyState[0]['monthlyPeak'] = 8

    ## Here we are converting the optimization time to the arrival time
    hr = optTime.hour
    minute =  optTime.minute / 60

    arrival_time = hr + minute
    duration_hour = 3
    e_need = 6
    delta_t = 0.25

    event = {
        "time": int(hr/ delta_t), # Hour or Arrival_hour?
        "e_need": e_need,
        "duration": int(duration_hour / delta_t),
        "station_pow_max": 6.6,
        "user_power_rate": 6,
        "limit_reg_with_sch": False,
        "limit_sch_with_constant": False,
        "sch_limit": 0,
        "historical_peak": 2
    }
    
    # Converting from the stateRecord to station info
    # Getting rid of the timestamp index, converting to kW
    sessionRecord = data_format_convertion(emptyState, hr, delta_t)
    
    par = Parameters(z0 = np.array([20, 30, 1, 1]).reshape(4, 1),
                    Ts = delta_t,
                    eff = 1.0,
                    soft_v_eta = 1e-4,
                    opt_eps = 0.0001,
                    TOU = TOU_15min(),
                    demand_charge_cost=500)

    prb = Problem(par, sessionRecord, hr,event=event)
    obj = Optimization_station(par, prb, hr)

    station_info, res = obj.run_opt(solution_algorithm = "grid_search")

    print(res['z_hr'])
    print(res['J'])


    prb = Problem(par, sessionRecord, hr,event=event)
    obj = Optimization_station(par, prb, hr)

    station_info, res = obj.run_opt(solution_algorithm = "BCD")

    # print(res['new_peak_sch'])
    # print(res['new_peak_reg'])

    return None


if __name__ == "__main__":
    main()
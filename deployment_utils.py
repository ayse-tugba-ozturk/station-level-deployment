from optimizer_station_V2 import Parameters, Problem, Optimization_station
import boto3
from boto3.dynamodb.conditions import Key, Attr
import time
import pandas as pd
import numpy as np
import datetime
import copy
from collections import defaultdict
import json
import os
import pytz

from decimal import Decimal
import random

def currentTime15min():
    # Get current time
    # pacific = pytz.timezone('US/Pacific')
    now = datetime.datetime.now()

    # Round down to the nearest 15 minutes
    rounded = now - datetime.timedelta(minutes=now.minute % 15, seconds=now.second, microseconds=now.microsecond)

    # print(f"Current Time: {now}")
    # print(f"Rounded Down to Nearest 15 Minutes: {rounded}")
    return rounded


def TOU_15min():
    """  units of cents / kwh
    """

    # We define the timesteps in the APP as 15 minute 
    delta_t = 0.25 #hour 
    # print("For delta_t: ",delta_t, "max number of intervals:",24/delta_t)
    ################## Define the TOU Cost ##########################################
    ## the TOU cost is defined considering the delta_t above, if not code raises an error.##

    # off-peak 0.175  cents / kwh 
    TOU_tariff = np.ones((96,)) * 17.5
    ## 4 pm - 9 pm peak 0.367 cents / kwh 
    TOU_tariff[64:84] = 36.7
    ## 9 am - 2 pm super off-peak 0.49 $ / kWh  to cents / kwh

    TOU_tariff[36:56] = 14.9
    return TOU_tariff

def unixTime(timeStamp):
    try:
        ts = int(pd.to_datetime(timeStamp, unit='s').timestamp())
    except ValueError: 
        try:
            timeStamp = pd.Timestamp(timeStamp)
        except TypeError:
            timeStamp = pd.Timestamp(int(timeStamp))


        ts = int(pd.to_datetime(timeStamp, unit='s').timestamp())
    return ts



def generateSessions(s_year, s_month, s_day, s_hour,
                     e_year, e_month, e_day, e_hour,
                     write_output = False, 
                     all_data = True,
                     path = "/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/data/",
                     f_name = "sessions.csv",
                     read_from_file = False):
    
    # TODO: If sessions exists

    if read_from_file:
        session_df = pd.read_csv("/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/data/"+"sessions.csv")

    else:
    
        # Obtain secret key for AWS
        txt_path = '/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/src/'
        f = open(txt_path +"secret.txt")
        text = f.readlines()

        # Access dynamodb on AWS
        dynamodb = boto3.resource(
            "dynamodb",
            aws_access_key_id=text[0][:-1],
            aws_secret_access_key=text[1][:-1],
            region_name="us-east-2")

        # Scan through all Sessions2 data to get session items
        table = dynamodb.Table('Sessions2')
        params = {'ProjectionExpression': "dcosId,userId,vehicle_model,vehicle_maxChgRate_W,siteId,stationId,connectTime,startChargeTime,Deadline,energyReq_Wh,estCost,reg_centsPerHr,sch_centsPerHr,sch_centsPerKwh,sch_centsPerOverstayHr,#Dur,DurationHrs,choice,regular,scheduled,cumEnergy_Wh,peakPower_W,power,lastUpdate",
                'ExpressionAttributeNames': {"#Dur":"Duration"}}

        # Repeat scan until LastEvaluatedKey is None
        start = time.time()
        done = False
        start_key = None
        temp = []
        while not done:
            if start_key:
                params['ExclusiveStartKey'] = start_key
            response = table.scan(**params)
            temp.extend(response.get('Items', []))
            print("Length of Scanned Items is {0} items".format(len(temp)))
            start_key = response.get('LastEvaluatedKey', None)
            done = start_key is None

        # Print elapsed time
        end = time.time()
        elapsed_time_min = np.floor((end-start)/60)
        elapsed_time_sec = (end-start) % 60
        elapsed_time = "Elapsed time: %d minutes, %d seconds" % (elapsed_time_min, elapsed_time_sec)
        print(elapsed_time)

        # Create dataframe
        session_df = pd.DataFrame(temp) 
        del temp

        session_df['connectTime']  = pd.to_datetime(session_df['connectTime'] )
        session_df['startChargeTime']  = pd.to_datetime(session_df['startChargeTime'] )
        session_df['Deadline']  = pd.to_datetime(session_df['Deadline'] )
        session_df['lastUpdate']  = pd.to_datetime(session_df['lastUpdate'])
        session_df = session_df.sort_values(by='connectTime')
        # session_df = session_df[session_df['siteId'] == 25]
        # session_df['interArrivalTime_min'] = session_df['connectTime'].diff().dt.seconds / 60
        plot_df = session_df[session_df['connectTime'].dt.year >= 2022]
        session_df['arrHour']=session_df['connectTime'].dt.hour

        session_df['cumEnergy_Wh']=session_df['cumEnergy_Wh'].astype(float)
        session_df['cumEnergy_KWh']=session_df['cumEnergy_Wh']/1000
        session_df['DurationHrs'] = session_df['DurationHrs'].astype(float)

        # 1. peakPower_W = 0 and cumEnergy_W = 0: delete  
        session_df = session_df[(session_df["peakPower_W"]!=0) & (session_df["cumEnergy_Wh"]!=0)]

        # 2. Fix the user peak_powers 
        # some users have historical peak power > 6.6 / 7 
        user_df = session_df[['userId','peakPower_W','vehicle_maxChgRate_W']].groupby('userId').max()
        user_df['session_counts'] = session_df[['userId','peakPower_W']].groupby('userId').count()

        validate_users = user_df[(user_df['session_counts']==1) & (user_df['peakPower_W']<=6000) ].index.to_list()
        user_df.reset_index(drop=False,inplace=True)

        user_df.rename(columns={"peakPower_W":'historical_peakPower_w'},inplace=True)
        session_df = session_df.merge(user_df[['userId','historical_peakPower_w']])


        session_df['endTime'] = session_df['startChargeTime'] + pd.to_timedelta(session_df['Duration'])
        cols = ['connectTime','choice','power','endTime','Duration', 'userId', 'Deadline', 'startChargeTime','DurationHrs','dcosId', 'lastUpdate' ]


        high_power_idx = list(session_df[session_df['historical_peakPower_w'] >= 5000].index)
        low_power_idx = list(session_df[session_df['historical_peakPower_w'] < 5000].index)

        session_df['highPower'] = pd.Series(dtype=int)

        session_df.loc[high_power_idx ,'highPower'] = 1
        session_df.loc[low_power_idx ,'highPower'] = 0

        session_df.loc[high_power_idx ,'userPower_kW'] = 6.6
        session_df.loc[low_power_idx ,'userPower_kW'] = 3.3

    if not all_data:
        session_df['connectTime']  = pd.to_datetime(session_df['connectTime'] )
        start_time = pd.Timestamp(year = s_year, month = s_month, day= s_day, hour = s_hour )
        end_time = pd.Timestamp(year = e_year, month = e_month, day= e_day, hour = e_hour )
        
        print("From {} till {} (incl.)".format(start_time, end_time))
        session_df = session_df[(session_df['connectTime']>=start_time) & 
               (session_df['connectTime']<=end_time)].sort_values(by='connectTime')


    if write_output:
        if ".csv" in f_name:
            session_df.to_csv(path +f_name,index=True)
        if ".pkl" in f_name:
            session_df.to_pickle(path +f_name)


    return session_df

def generateExpectedDemand(session_df,
                           write_output = False,
                            path = "/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/data/",
                            f_name = "expectedDemand.csv"):
    
    iterables = [[0,1], range(0,24)]
    idx = pd.MultiIndex.from_product(iterables, names=['highPower','arrHour'])
    # pd.DataFrame(index=idx,columns = ['DurationHrs', 'cumEnergy_Wh', 'interArrivalTime_min', 'arrivalHour','cumEnergy_KWh', 'count'])
    expectedDemand = session_df.groupby(['highPower','arrHour']).mean()
    expectedDemand['count'] = session_df.groupby(['highPower','arrHour']).count()['dcosId']
    expectedDemand = expectedDemand.reindex(idx).reset_index(drop=False)
    expectedDemand.rename(columns = {'cumEnergy_Wh':"estEnergyWh", 
                                    'cumEnergy_KWh':"estEnergykWh", 
                                    "DurationHrs":"estDurationHrs"},inplace=True)
    
    expectedDemand['count'].fillna(0,inplace=True)
    expectedDemand.fillna(method='ffill',inplace=True)

    if write_output:
        if ".csv" in f_name:
            expectedDemand.to_csv(path +f_name)
        if ".pkl" in f_name:
            expectedDemand.to_pickle(path +f_name)
    return expectedDemand

def generateBackupPrices(expectedDemand,
                        timestamp = pd.Timestamp(year = 2023, month = 9, day=16, hour=0, minute=0)):
    
    """ Generates the expected demand table and optimal solution for NO VEHICLE Case
        Think about the very first user """

    # Think about the very first user, everything should be ready for that user

    emptyState = emptyStateRecord(timestamp)
    States, optimizedPrices_V3 = generateOptPricePower(emptyState, expectedDemand, timestamp, 0, opt_horizon=24, TOU_tariff=TOU_15min())
    
    # print(States)
    return States, optimizedPrices_V3

def dummyUserPower(choice):
    
    """ Returns Array but this can also be a dictionary """

    ### Yifei: In the optimizer, we require strictly that HIGH power corresponds to 6.6 kW and LOW power corresponds to 3.3 kW.
    ### Aka, for REG, we require the first several intervals to be 6.6 kW (must satisfy, cannot be other values like 6, 6, 6...)
    
    ts = pd.date_range(start = pd.Timestamp(2023, 3, 6, 7, 45, 0), periods=10, freq="15min")
    ts = [unixTime(t) for t in ts]
    
    p0 = 6600
    p1 = 6600
    p2 = 6600
    p3 = 6600
    p4 = 6600
    p5 = 3300

    if choice == "REG":
        
        ## Can you clarify up to what are we recording for the REG option? Is it the N_ASAP? 

        ## Yifei: Yes, the length of REG_powers should be the same with N_ASAP.
        powers = np.array([[ts[0], p0], 
                           [ts[1], p1], 
                           [ts[2], p2], 
                           [ts[3], p3], 
                           [ts[4], p4],
                           [ts[5], 3300]])

    elif choice == "SCH":
        powers = np.array([ [ts[0], 0], [ts[1], p1], [ts[2], p2], [ts[3], p3], [ts[4], p4],
                           [ts[5], 6600], [ts[6], 0], [ts[7], 0], [ts[8], 0], [ts[9], 0]])
    else:
        powers = np.zeros(shape=(6,2))
    return powers

def dummyStateRecord():
    """ 
    Returns: List of Dictionaries
    Each entry is a state dictionary with keys: 
    
    monthlyPeak, int
    timeStamp, int
    sessions, list of dictionaries with keys
    dcosId, int
    choice, string
    powerRate, string
    energyNeeded, float
    deadline, int
    optPower, array 
        """
    
    stateRecord = [
        {"monthlyPeak":18, 
         "timestamp":unixTime(pd.Timestamp(2023, 3, 6, 7, 45, 0)), ## Last record TS(decision of the last vehicle)
         "sessions": [ 
             {
                "dcosId" : 1001,
                "choice": "SCH",
                "powerRate": "HIGH",
                "energyNeeded" : 8250,
                "deadline" : unixTime(pd.Timestamp(2023, 3, 6, 9, 45, 0)), # So here the values are all discretized to 15 min intervals?
                "optPower" : dummyUserPower("SCH") 
             }, 
             {
                "dcosId" : 1002,
                "choice": "REG",
                "powerRate": "HIGH",
                "energyNeeded" : 9075,
                "deadline" : unixTime(pd.Timestamp(2023, 3, 6, 10, 0, 0)),
                "optPower" : dummyUserPower("REG") 
             }, 
         ]
        }
    ]
    print(stateRecord[0]["sessions"][0]["optPower"][:,1].sum() * 0.25)
#     assert  == 3750
    
    return stateRecord

def arrHourList(arrHour, optHorizon):
    """ arrHour, int: current optimization hour
        optHorizon, int: how long to optimize in hours """ 
    if arrHour <= (24-optHorizon):
        overnight = False
        return list(range(arrHour,(arrHour+optHorizon))), overnight
    else: 
        lst = list(range(arrHour,24))
        # lst.extend(list(range(0 , ((arrHour+optHorizon)-24) )))
        print("Overnight charging")
        overnight = True
        ## Overnight charging: like [23, 0, 1, 2]

        return lst, overnight
    
def emptyStateRecord(timestamp):
    """ 
    Input:
    timestamp: Choice of timestamp for the very last optimization output state

    Returns: List of Dictionaries
    Each entry is a state dictionary with keys: 

    monthlyPeak, int
    timestamp, int
    sessions, list of dictionaries with keys
    dcosId, int
    choice, string
    powerRate, string
    energyNeeded, float
    deadline, int
    optPower, array 
        """
    
    # TODO 
    stateRecord = [
        {"monthlyPeak":0, ###### It should get the monthly Peak 
            "timestamp":unixTime(timestamp), ## Last record TS(decision of the last vehicle)
            "sessions": [ ]
        }
    ]

    return stateRecord

def data_format_convertion(stateRecords, opt_hour, delta_t = 0.25):

    """ Yifei needed this, not sure how are we using"""
    timezone = datetime.timezone(datetime.timedelta(hours=0))
    stateRecord = copy.deepcopy(stateRecords[0]["sessions"])
    if not stateRecord:
        return None
    # num_users = len(stateRecord)
    res = []
    for user in stateRecord:
        try: 
            user["optPower"] = user["optPower"].tolist()
        except AttributeError: 
            user["optPower"] = user["optPower"]
            
        user["power_rate"] = 6.6 if user["powerRate"] == "HIGH" else 3.3

        start_time_obj = datetime.datetime.fromtimestamp(int(user["optPower"][0][0]), timezone) # the timestamp of the first time slot
        user["start_time"] = float(start_time_obj.hour + start_time_obj.minute / 60)

        end_time_obj = datetime.datetime.fromtimestamp(int(user["optPower"][-1][0]), timezone) # Or retrieve the last time slot??
        user["end_time"] = float(end_time_obj.hour + end_time_obj.minute / 60) + delta_t
        if user["end_time"] <= opt_hour:
            continue

        user["optPower"] = np.round(np.array([x[1] for x in user["optPower"]]) / 1000, 2)
        user["price"] = 25 if user["choice"] == "SCH" else 30
        user["energyNeeded"] = float(user["energyNeeded"]) / 1000
        del(user["deadline"])
        res.append(user)

    return res

def convertOutput(stateRecords, station_info, res, hour, user_choice, optTime, delta_t = 0.25):
    """ Convert the output to the original format """
    # print("optTime)
    # Whats the difference of stateRecords and station_info? *******8
    new_state = copy.deepcopy(stateRecords[0])

    new_state["monthlyPeak"] = round(res["new_peak_sch"][0], 2) if user_choice == "SCH" else round(res["new_peak_reg"][0], 2)

    # Update the timestamp. Tugba: What should be the timestamp here?
    # Are we assuming always minute = 0? 
    # We can directly assign this to optTime maybe 
    # print(optTime)
    # if optTime.hour > hour: # If it is overnight.
    #     new_TimeStamp = optTime + datetime.timedelta(days=1)
    #     # print(new_TimeStamp)
    #     new_TimeStamp = new_TimeStamp.replace(hour=hour, minute=0, second=0)
    #     # print(new_TimeStamp)
    # else:
    #     new_TimeStamp = optTime.replace(hour=hour, minute=0, second=0)
    #     # print(new_TimeStamp)
    new_state["timestamp"] = unixTime(optTime) # Correct timestamp 
    print(pd.to_datetime(new_state["timestamp"],unit='s'))
    

    # new_state["timestamp"] = unixTime(pd.Timestamp(2023, 3, 6, hour, 0 , 0))
    finishing_list = []
    if new_state["sessions"]:
        for i, user in enumerate(new_state["sessions"]):
            # Changed this because we need an array for code to not break.. 
            # We can agree on a convention later
            # df = pd.DataFrame(user["optPower"])
            

            # df[0] = pd.to_datetime(df[0], unit='s')

            user["optPower"] = recover_json_serialized_power_array(user["optPower"])

            # print(df)
            timezone = datetime.timezone(datetime.timedelta(hours=0))
            end_time_obj = datetime.datetime.fromtimestamp(int(user["optPower"][-1][0]), timezone) # Or retrieve the last time slot??

            end_time = float(end_time_obj.hour + end_time_obj.minute / 60) + delta_t


            if end_time <= hour:
                finishing_list.append(i)
                continue
            user_update = [d for d in station_info if d["dcosId"] == user["dcosId"]][0] # The updated user info from opt output
            TOU_idx = user_update["TOU_idx"]
            try:
                user["optPower"][TOU_idx:, 1] = (np.ceil(user_update["optPower"][TOU_idx:] * 1000)).astype(int) # Retaining the UNIXTIME and updating the power
            except ValueError:
                print('TOU_idx: ',TOU_idx)
                print('user update:',user_update)
                print("user[optPower]:",user["optPower"],
                      user["optPower"][TOU_idx:, 1])
                print("user_update[optPower]:",user_update["optPower"],"\n",(np.ceil(user_update["optPower"][TOU_idx:] * 1000)).astype(int))
                user["optPower"][TOU_idx:, 1] = (np.ceil(user_update["optPower"][TOU_idx:] * 1000)).astype(int)


    if finishing_list:
        # Remove the index in finishing_list from new_state["sessions"]
        new_state["sessions"] = [user for i, user in enumerate(new_state["sessions"]) if i not in finishing_list]
    new_user = dict()
    # new_user["dcosId"] = "dummyUser"

    new_user["dcosId"] = -9999
    new_user["choice"] = user_choice  # This choice and OPT power / price are decided outside the optimizer
    new_user["powerRate"] = "HIGH" if res["power_rate"] >= 5 else "LOW"
    new_user["energyNeeded"] = int(1000 * res['e_need'])
    new_user["optPower"], _ , new_user["deadline"] = powerOutput(res, user_choice, optTime)
    # print(new_user["deadline"])
    ## TO-DO: How to get the deadline? Is it the rounded and discretized time or the actual time? For example, 8:45(Rounded) / 8:47(Actual)?
    # new_state["utilityCost"] = res["utility_cost_sch"] if user_choice == "SCH" else res["utility_cost_reg"]
    new_state["sessions"].append(new_user)

    return new_state

def powerOutput(res, user_choice, optTime, delta_t = 0.25):
    """ Convert the output to the original format """
    start_timestamp = res["time_start"] * delta_t
    start_timestamp_hour = int(start_timestamp)
    start_timestamp_minute = int((start_timestamp % 1) * 60)
    start_time = optTime.replace(hour=start_timestamp_hour, minute=start_timestamp_minute, second=0) # Start_time == optTime here.
    # start_time = datetime.datetime(2023, 3, 6, start_timestamp_hour, start_timestamp_minute, 0)
    t0 = unixTime(start_time)

    end_timestamp = res["time_end_SCH"] * delta_t if user_choice == "SCH" else res["time_end_REG"] * delta_t
    end_timestamp_hour = int(end_timestamp)
    end_timestamp_minute = int((end_timestamp % 1) * 60)
    if end_timestamp_hour >= 24:
        end_timestamp_hour -= 24
        end_time = optTime + datetime.timedelta(days=1)
        end_time = end_time.replace(hour=end_timestamp_hour, minute=end_timestamp_minute, second=0)
    else:
        end_time = optTime.replace(hour=end_timestamp_hour, minute=end_timestamp_minute, second=0)
    # end_time = datetime.datetime(2023, 3, 6, end_timestamp_hour, end_timestamp_minute, 0)
    t1 = unixTime(end_time)

    timestamps = np.arange(t0, t1, delta_t * 60 * 60).astype(int)   # In seconds, for example, 0.25 * 60 = 15min in seconds
    optPower = copy.deepcopy(res["sch_powers"]) * 1000 if user_choice == "SCH" else copy.deepcopy(res["reg_powers"]) * 1000
    optPower = optPower.astype(int)
    output_power = np.concatenate((timestamps.reshape(-1, 1), optPower.reshape(-1, 1)), axis=1)

    ## Here the max_timestamp is used to calculate the aggregated power(decide which interval has power)
    max_timestamp = res["time_max"] * delta_t
    max_timestamp_hour = int(max_timestamp)
    max_timestamp_minute = int((max_timestamp % 1) * 60)
    if max_timestamp_hour >= 24:
        max_timestamp_hour -= 24
        max_time = optTime + datetime.timedelta(days=1)
        max_time = max_time.replace(hour=max_timestamp_hour, minute=max_timestamp_minute, second=0)
    else:
        max_time = optTime.replace(hour=max_timestamp_hour, minute=max_timestamp_minute, second=0)
    # max_time = datetime.datetime(2023, 3, 6, max_timestamp_hour, max_timestamp_minute, 0)
    t2 = unixTime(max_time)

    timestamps_agg = np.arange(t0, t2, delta_t * 60 * 60).astype(int)
    aggPower = copy.deepcopy(res["sch_agg"]) * 1000 if user_choice == "SCH" else copy.deepcopy(res["reg_agg"]) * 1000
    aggPower = aggPower.astype(int)
    output_power_agg = np.concatenate((timestamps_agg.reshape(-1, 1), aggPower.reshape(-1, 1)), axis=1)

    return output_power, output_power_agg, t1



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def recover_json_serialized_power_array(json_serialized_item):
    return np.asarray(json_serialized_item)


def generateOptPricePowerFromDummyInput(expected_demand, optTime):
    
    unixOptTime = unixTime(optTime)  ## Convert the optTime to unix time
    arrival_hour = optTime.hour

    # We only take the next 4 hours of the price table

    ## read the stateRecords last entry
    stateRecords = dummyStateRecord()
    stateRecords[0]["sessions"] = list()  ### Temperarily set the sessions to be empty!!!!

    try:
        expected_demand.set_index(['arrHour','highPower'], inplace=True)
    except KeyError:
        print("")
        
    States, expectedDemand = generateOptPricePower(stateRecords, expected_demand, optTime, arrival_hour)
        
    return States, expectedDemand



def generateOptPricePower(stateRecords, 
                          expected_demand,
                          optTime,
                          arrival_hour,
                          opt_horizon=4,
                          delta_t = 0.25,
                          TOU_tariff = np.ones((96,1))):
    """
    Generate the expected demand price table
    optTime: The time when the re-optimization is triggered. pd.Timestamp.
    arrival_hour: The hour of optTime
    """
    
    try:
        expected_demand.set_index(['arrHour','highPower'], inplace=True)
    except KeyError:
        print("")

    States = defaultdict(dict)
    optHours, overnight = arrHourList(arrival_hour, optHorizon=opt_horizon)

    ## read the expected demand table
    for highPower in [0,1]:
        for hour in optHours:

            ## Here we are converting the optimization time to the arrival time
            hr = optTime.hour
            minute =  optTime.minute / 60

            arrival_time = hr + minute
            duration_hour = expected_demand.loc[(hour, highPower), 'estDurationHrs']
            e_need = expected_demand.loc[(hour, highPower), 'estEnergykWh']

            event = {
                "time": int(hour / delta_t), # Hour or Arrival_hour?
                "e_need": e_need,
                "duration": int(duration_hour / delta_t),
                "station_pow_max": 6.6,
                "user_power_rate": expected_demand.loc[(hour, highPower), 'userPower_kW'],
                "limit_reg_with_sch": False,
                "limit_sch_with_constant": False,
                "sch_limit": 0,
                "historical_peak": stateRecords[0]['monthlyPeak']
            }
            
            # Converting from the stateRecord to station info
            # Getting rid of the timestamp index, converting to kW
            sessionRecord = data_format_convertion(stateRecords, hour, delta_t)
            
            par = Parameters(z0 = np.array([25, 30, 1, 1]).reshape(4, 1),
                         Ts = delta_t,
                         eff = 1.0,
                         soft_v_eta = 1e-4,
                         opt_eps = 0.0001,
                         TOU = TOU_15min(),
                         demand_charge_cost=18)

            prb = Problem(par=par, event=event)
            try: 
                obj = Optimization_station(par, prb, sessionRecord, hour)
                station_info, res = obj.run_opt(solution_algorithm = "BCD")

                States["hour" + str(hour) + "-" + str(highPower)]["SCH"] = convertOutput(stateRecords, station_info, res, hour, "SCH", optTime)
                States["hour" + str(hour) + "-" + str(highPower)]["REG"] = convertOutput(stateRecords, station_info, res, hour, "REG", optTime)

                ### Heyy so this is for US to be able to test. We won't do this type of output in the server
                # However we will send the Right hand side of the below lines.
                # RHS should be a state record, in the server it will be indexed by hour and power and choice
                expected_demand.loc[(hour, highPower), 'SCH_expected_power_W']= json.dumps(States["hour" + str(hour) + "-" + str(highPower)]['SCH'],
                                                                                    cls=NumpyEncoder)

                expected_demand.loc[(hour, highPower), 'REG_expected_power_W']= json.dumps(States["hour" + str(hour) + "-" + str(highPower)]['REG'],
                                                                                    cls=NumpyEncoder)
                # if dcosId==2062:
                #     print(States)

                ### How do we update the expected demand table? Make a slice for 4 hours or take the whole?
                reg_centsPerHr, sch_centsPerHr = res["reg_centsPerHr"], res['sch_centsPerHr']
                expected_demand.loc[(hour, highPower), 'sch_centsPerHr'] = sch_centsPerHr
                expected_demand.loc[(hour, highPower), 'reg_centsPerHr'] = reg_centsPerHr

            except: 
                print("Unable to solve")
                States["hour" + str(hour) + "-" + str(highPower)]["SCH"] = None
                States["hour" + str(hour) + "-" + str(highPower)]["REG"] = None
                ### Heyy so this is for US to be able to test. We won't do this type of output in the server
                # However we will send the Right hand side of the below lines.
                # RHS should be a state record, in the server it will be indexed by hour and power and choice
                expected_demand.loc[(hour, highPower), 'SCH_expected_power_W']= None

                expected_demand.loc[(hour, highPower), 'REG_expected_power_W']= None
                # if dcosId==2062:
                #     print(States)

                ### How do we update the expected demand table? Make a slice for 4 hours or take the whole?
                # reg_centsPerHr, sch_centsPerHr = res["reg_centsPerHr"], res['sch_centsPerHr']
                expected_demand.loc[(hour, highPower), 'sch_centsPerHr'] = None
                expected_demand.loc[(hour, highPower), 'reg_centsPerHr'] = None
                


    return States, expected_demand

def get_new_state(expected_demand, new_session_start):
    new_State = expected_demand.loc[(new_session_start.hour, 0),
                                   "SCH_expected_power_W"]
    new_State = json.loads(new_State)

    for session in new_State['sessions']:
        if session['dcosId'] == -9999 :
            new_State['sessions'].remove(session)

        if session['deadline'] <= unixTime(new_session_start):
            print(pd.to_datetime(session['deadline'])," <= ",unixTime(new_session_start))
            print("removed session:",session['dcosId']) ## Check if it is removing correct or wrong session!!
            new_State['sessions'].remove(session)
    return new_State

def update_demand_table(last_opt_time, 
                        path_demand_table):
    
    """ 
    Updates the time fields of the States in date-change
    Need to control for daylight savings etc. 
    """
    new_expected_demand = pd.read_csv(path_demand_table)
    # high power, low power, REG, SCH 
    for idx in range(len(new_expected_demand.index)):
        for choice in ['REG','SCH']:

            new_State  = json.loads(new_expected_demand.loc[idx, 
                                                            "{}_expected_power_W".format(choice)]
                                                            )
            

            # print("Changed from:", pd.to_datetime(new_State ['timestamp'], unit='s'))
            new_State['timestamp'] += (24 * 60 * 60)
            # print("Changed to:",pd.to_datetime(new_State['timestamp'], unit='s'))
            for i, session in enumerate(new_State['sessions']):
                old_opt_power_array = np.array(session['optPower'])
                # print(pd.to_datetime(old_opt_power_array[0,0],unit='s'))
                old_opt_power_array[:,0] += (24 * 60 * 60)
                # print(pd.to_datetime(old_opt_power_array[0,0],unit='s'))
                new_State['sessions'][i]['optPower'] = old_opt_power_array.tolist()
                new_State['sessions'][i]['deadline'] += (24 * 60 * 60)
        
            new_expected_demand.loc[idx, "{}_expected_power_W".format(choice)] = json.dumps(new_State,cls=NumpyEncoder)


    return new_expected_demand



def calculate_probability(z_reg, z_sch):
    u_reg =  0.3411 -0.0184*(z_sch - z_reg)*.5
    u_sch = 0.0184*(z_sch - z_reg)*.5
    # u_leave = -1. + 0.005*(np.mean([z_sch, z_reg]))

#     print(u_leave, u_sch, u_reg)
    denom = np.exp(u_reg) + np.exp(u_sch)

    p_reg = np.exp(u_reg) / denom
    p_sch = np.exp(u_sch) / denom
    # p_leave = np.exp(u_leave) / denom

#     print(p_reg, p_sch, p_leave, np.sum([p_reg, p_sch, p_leave]))
    return p_reg, p_sch


def monte_carlo_sim_choice(REG, SCH):


    # Define the possible outcomes and their probabilities
    outcomes = ['REG', 'SCH']

    probabilities = list(calculate_probability(REG, SCH))

    # Generate a binary random variable with a probability of success of 0.3
    result = random.choices(outcomes, weights=probabilities)[0]
    
    return result


def simulate_station(count = 100):
    np.random.seed(seed=100)
    for i in range(1,10):
        session_df = pd.read_csv("/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/data/" +  "sessions_baselineI.csv")
        print("Optimizing {} sessions".format(len(session_df)))
        session_df['connectTime']  = pd.to_datetime(session_df['connectTime'] )

        path_demand_table = "/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/data/" +  "expectedDemand_baseline.csv"
        expected_demand = pd.read_csv(path_demand_table)
        
        last_opt_time = pd.Timestamp(year =2022, month=3,day =14, hour = 5)

        for row in session_df.index:

            new_session_start = session_df.loc[row,"connectTime"]
            new_session_id = session_df.loc[row,"dcosId"]
            new_session_power = session_df.loc[row,"highPower"]
            arrHour = new_session_start.hour

            print("")
            print("Now we are optimizing for row:",row,"at time:",session_df.loc[row,"connectTime"])
            print("New User DcosId: {}, Power: {}".format(new_session_id, new_session_power))
            print("")


            # if new_session_id == 2062:
            #     print(new_State['sessions'])
            # WE CAN ROUND UPTO THE NEAREST 15MIN TOO 
            curr_opt_time = new_session_start.replace(minute=0, second=0)
            opt_horizon = 4

            # IF OPTIMIZING FOR 4 HOURS HORIZON THAN THIS MAY STILL FAIL
            # ALSO CHOICE OF NEW_LAST_OPT TIME CAN BE IMPROVED 
            # HERE I AM TRYING TO MAKE IT A BIT MORE FASTER BY AVOIDING OVERNIGHT REOPTIMIZATION
            # ASSUMING NO OVERNIGHT VEHICLES AT THE MOMENT, IF THAT HAPPENS WE WILL IMPROVE
            if last_opt_time.day!= curr_opt_time.day:
                print("Changing OPTIMIZATION DAY from {}".format(last_opt_time))
                try:
                    last_opt_time = pd.Timestamp(year = curr_opt_time.year, 
                                                month= curr_opt_time.month,
                                                day = curr_opt_time.day, 
                                                hour = curr_opt_time.hour - 1)
                except: 
                    last_opt_time = pd.Timestamp(year = curr_opt_time.year, 
                                                month= curr_opt_time.month,
                                                day = curr_opt_time.day, 
                                                hour = curr_opt_time.hour,
                                                minute=0)
                    

                print("TO: {}".format(last_opt_time))
                ## FUNCTION TO GENERATE EXPECTED DEMAND FOR THE NEW DAY 

                ## Maybe check here, if there are still vehicles in the station 
                expected_demand = update_demand_table(last_opt_time, path_demand_table)
                
                
            while curr_opt_time - last_opt_time > pd.Timedelta(hours=opt_horizon):
                print("Current optimization:",curr_opt_time, "Last optimization:", last_opt_time)
                print("Haven't optimized for a while, re-optimize")
                new_State = get_new_state(expected_demand, last_opt_time)
                last_opt_time += pd.Timedelta(hours=opt_horizon) # Re-optimize every 4 hours
                stateRecords = [new_State] # States at the last opt hour
                States, expected_demand = generateOptPricePower(stateRecords,
                                    expected_demand,
                                    last_opt_time,
                                    last_opt_time.hour)
            
            arrHour = new_session_start.hour
            
            #### WE CHECK THE TABLE, GET THE NEW STATE CORRESPONDING TO THE NEW USER CHOICE

            try:
                expected_demand.set_index(['arrHour','highPower'], inplace=True)
            except KeyError:
                print("")

            Z_SCH  = expected_demand.loc[(new_session_start.hour, new_session_power),
                                                                'sch_centsPerHr']
            
            Z_REG = expected_demand.loc[(new_session_start.hour, new_session_power),
                                                                'reg_centsPerHr']
            
            new_session_choice = monte_carlo_sim_choice(Z_REG, Z_SCH)
            print('New Session Choice:', new_session_choice)

            new_State = expected_demand.loc[(new_session_start.hour, new_session_power),
                                            new_session_choice[:3]+ "_expected_power_W"]
            
            new_State = json.loads(new_State)

            session_df.loc[row,'sch_centsPerHr_opt'] = Z_SCH
            session_df.loc[row,'reg_centsPerHr_opt'] = Z_REG
            session_df.loc[row,'choice_sim'] = new_session_choice 
            

            # TODO: EXPECTED DEMAND IS ALSO INDEXED BY TIMESTAMP, 
            # SO THE TIMESTAMPS SHOULD MATCH TO NOT DROP
            for session in new_State['sessions']:
                if session['dcosId'] == -9999:
                    print("Changed dummy to {}".format(new_session_id))
                    session['dcosId'] = int(new_session_id)
                # TODO: Check if the TIMESTAMPS are correct
            for session in new_State['sessions']:
                if session['deadline'] <= unixTime(new_session_start):
                    print(pd.to_datetime(session['deadline'],unit='s')," <= ",new_session_start)
                    print("removed session:",session['dcosId']) ## Check if it is removing correct or wrong session!!
                    new_State['sessions'].remove(session)

            print("On-going sessions:", len(new_State['sessions']))
            print("New Choice:",new_session_choice)
            session_df.loc[row,'New State'] = json.dumps(new_State,cls=NumpyEncoder)
            session_df.to_pickle("/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/data/OptimizationTests/" +  "sessions_baselineI_optimized_sim_{}.pkl".format(i))
            # expected_demand.to_pickle("/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/data/OptimizationTests/" +  "expected_demand_{}.pkl".format(row))
            

            ### Identify the users at the Station
            optTime = new_session_start

            unixOptTime = unixTime(optTime)
            arrival_hour = optTime.hour

            stateRecords = [new_State]


            States, expected_demand= generateOptPricePower(stateRecords, 
                                    expected_demand,
                                    optTime, 
                                    arrival_hour)
        #     for hour in optHours:
        #         for power in [0, 1]:
        #             save_fig(new_session_id, hour, power)
            last_opt_time = optTime





def table_to_optimizedStates(table):
    # Dictionary to store the data
    optimizedStates = {}

    # Scan the table
    response = table.scan()

    # Handle Decimal types coming from DynamoDB
    def decimal_default(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError

    # Process scanned items
    for item in response['Items']:
        arrHour_highPower = item['arrHour_highPower']
        choice = item['choice']

        # Initialize if key does not exist
        if arrHour_highPower not in optimizedStates:
            optimizedStates[arrHour_highPower] = {}

        # Populate data
        optimizedStates[arrHour_highPower][choice] = {
            'monthlyPeak': decimal_default(item['monthlyPeak']),
            'timestamp': int(item['timestamp']),
            'sessions': [
                {
                    'dcosId': int(session['dcosId']),
                    'choice': session['choice'],
                    'powerRate': session['powerRate'],
                    'energyNeeded': decimal_default(session['energyNeeded']),
                    'optPower': [[decimal_default(x) for x in pair] for pair in session['optPower']],
                    'deadline': int(session['deadline'])
                }
                for session in item['sessions']
            ]
        }
    return optimizedStates

def table_to_stateRecord(table):
    # Dictionary to store the data
    stateRecord = {}

    # Scan the table
    response = table.scan()

    # Handle Decimal types coming from DynamoDB
    def decimal_default(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError

    # Process scanned items
    for item in response['Items']:
        try: 
            timestamp = decimal_default(item['timestamp'])
        except KeyError:
            timestamp = decimal_default(item['timeStamp'])

        # Populate data
        stateRecord= {"timestamp":int(timestamp),
                    'monthlyPeak': decimal_default(item['monthlyPeak']),
                    'sessions': [
                {
                    'dcosId': int(session['dcosId']),
                    'choice': session['choice'],
                    'powerRate': session['powerRate'],
                    'energyNeeded': decimal_default(session['energyNeeded']),
                    'optPower': [[decimal_default(x) for x in pair] for pair in session['optPower']],
                    'deadline': int(session['deadline'])
                } for session in item['sessions']]
        }
    return stateRecord


def convert_floats_to_decimals(obj):
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = convert_floats_to_decimals(obj[i])
        return obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_floats_to_decimals(v)
        return obj
    elif isinstance(obj, float):
        return Decimal(str(obj))
    else:
        return obj

def stringify_keys(d):
    """Recursively convert all keys in a dictionary (and nested dictionaries) to strings."""
    if not isinstance(d, dict):
        return d
    return {str(k): stringify_keys(v) for k, v in d.items()}

def main():
    



    return None


if __name__ == "__main__":
    main()
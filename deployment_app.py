from deployment_utils import emptyStateRecord, arrHourList, data_format_convertion, unixTime, TOU_15min, convertOutput, NumpyEncoder,convert_floats_to_decimals, currentTime15min, table_to_optimizedStates, stringify_keys, table_to_stateRecord
from optimizer_station_V2 import Parameters, Problem, Optimization_station
# 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
import time
import pandas as pd
import numpy as np
import datetime
import copy
from collections import defaultdict
import json
import os
import random
from decimal import Decimal

# #### AWS FUNCTIONS ####
from fastapi import FastAPI, Response, status, Body, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import boto3, json, hashlib, time, random, pickle
from boto3.dynamodb.conditions import Key, Attr

app = FastAPI()

# Obtain secret key for AWS
# "txt_path = '/Users/aysetugbaozturk/Documents/eCal/SlrpEV/pev-digital-twin/src/'
# f = open(txt_path +"secret.txt")"
f = open("/home/s/sl/slrpev/SlrpEV_algorithms/secret.txt")
text = f.readlines()

dynamodb = boto3.resource(
    "dynamodb",
            aws_access_key_id=text[0][:-1],
            aws_secret_access_key=text[1][:-1],
    region_name="us-east-2",
)

DAY_SECONDS = 86400
FOUR_HR_SECONDS = 4*3600
NO_SITES = 1
WS_TO_KWH = 1/(1000*3600)
WH_TO_KWH = 1/1000

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generateOptPricePower(stateRecords, 
                          expected_demand,
                          optTime,
                          arrival_hour,   #### From the new user 
                          opt_horizon,
                          delta_t = 0.25,
                          TOU_tariff = np.ones((96,1))):
    
    """
    Generate the expected demand price table
    optTime: The time when the re-optimization is triggered. pd.Timestamp.
    arrival_hour: The hour of optTime
    expected_demand table from the server. We also save the optimized prices and power for each time we connect

    TODO: 
    make a copy of the expecteddemand call it optimizedPricesV3

    """
    print("opt horizon:", opt_horizon)
    try:
        expected_demand.set_index(['arrHour','highPower'], inplace=True)
    except KeyError:
        print("")

    

    ### optimizedStates 
    States = defaultdict(dict)
    optHours, overnight = arrHourList(arrival_hour, optHorizon=opt_horizon)

    ## read the expected demand table
    for highPower in [0,1]:
        for hour in optHours:
            print("Optimizing for:",hour,highPower)
            ## Here we are converting the optimization time to the arrival time
            hr = optTime.hour
            minute =  optTime.minute / 60

            arrival_time = hr + minute

            duration_hour = expected_demand.loc[(hour, highPower), 'estDurationHrs']
            e_need = expected_demand.loc[(hour, highPower), 'estEnergykWh']

            event = {
                "time": int(arrival_time/ delta_t), # Hour or Arrival_hour?
                "e_need": e_need,
                "duration": int(duration_hour / delta_t),
                "station_pow_max": 6.6,
                "user_power_rate": expected_demand.loc[(hour, highPower), 'userPower_kW'],
                "limit_reg_with_sch": False,
                "limit_sch_with_constant": False,
                "sch_limit": 0,
                "historical_peak": stateRecords[0]['monthlyPeak']
            }
            
            print("Peak: ",stateRecords[0]['monthlyPeak'])
            # Converting from the stateRecord to station info
            # Getting rid of the timestamp index, converting to kW
            sessionRecord = data_format_convertion(stateRecords, hour, delta_t)
            
            par = Parameters(z0 = np.array([20, 30, 1, 1]).reshape(4, 1),
                         Ts = delta_t,
                         eff = 1.0,
                         soft_v_eta = 1e-4,
                         opt_eps = 0.0001,
                         TOU = TOU_15min(),
                         demand_charge_cost=100 ) ## cents/ kW

            prb = Problem(par, sessionRecord, arrival_time,event=event)
            # try: 
            obj = Optimization_station(par, prb, arrival_time)
            station_info, res = obj.run_opt(solution_algorithm = "BCD")

            reg_centsPerHr, sch_centsPerHr = res["reg_centsPerHr"], res['sch_centsPerHr']
            
            States["hour" + str(hour) + "-" + str(highPower)]["SCH"] = convertOutput(stateRecords, station_info, res, hour, "SCH", optTime)
            States["hour" + str(hour) + "-" + str(highPower)]["REG"] = convertOutput(stateRecords, station_info, res, hour, "REG", optTime)
            States["hour" + str(hour) + "-" + str(highPower)]["SCH"]['reg_centsPerHr'] = reg_centsPerHr
            States["hour" + str(hour) + "-" + str(highPower)]["SCH"]['sch_centsPerHr'] = sch_centsPerHr
            States["hour" + str(hour) + "-" + str(highPower)]["REG"]['reg_centsPerHr'] = reg_centsPerHr
            States["hour" + str(hour) + "-" + str(highPower)]["REG"]['sch_centsPerHr'] = sch_centsPerHr
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
            
            expected_demand.loc[(hour, highPower), 'sch_centsPerHr'] = sch_centsPerHr
            expected_demand.loc[(hour, highPower), 'reg_centsPerHr'] = reg_centsPerHr

            # except: 
            #     print("Unable to solve")
            #     States["hour" + str(hour) + "-" + str(highPower)]["SCH"] = None
            #     States["hour" + str(hour) + "-" + str(highPower)]["REG"] = None
            #     ### Heyy so this is for US to be able to test. We won't do this type of output in the server
            #     # However we will send the Right hand side of the below lines.
            #     # RHS should be a state record, in the server it will be indexed by hour and power and choice
            #     expected_demand.loc[(hour, highPower), 'SCH_expected_power_W']= None

            #     expected_demand.loc[(hour, highPower), 'REG_expected_power_W']= None
            #     # if dcosId==2062:
            #     #     print(States)

            #     ### How do we update the expected demand table? Make a slice for 4 hours or take the whole?
            #     # reg_centsPerHr, sch_centsPerHr = res["reg_centsPerHr"], res['sch_centsPerHr']
            #     expected_demand.loc[(hour, highPower), 'sch_centsPerHr'] = None
            #     expected_demand.loc[(hour, highPower), 'reg_centsPerHr'] = None
                

    return States, expected_demand



def updateStateRecord(States, 
                      lastStateRecord,
                       arrHour, 
                       choice,
                       powerLevel, 
                       sessionId,
                       sessionStart,
                       connection_count_helper = 0):
    

    """ When user makes a selection 
    States: Optimized states from the last call of the getOptPricePower
    arrHour, int
    powerLevel, [0,1]
    choice, "SCH" or "REG" from the new user
    connection_count_helper: When we started to use the engineRecords, how we update the stateRecords is not applicable directly anymore.
    But changing the method requires a loot of changes. So we are using a helper count, 
    if the stateRecord is being updated for the first time with the given optimizedStates the code works the same. 
    If the stateRecord is already updated then we copy the dummy session 

    """
    if connection_count_helper <= 1:
        new_State = [States["hour{}-{}".format(arrHour, powerLevel)][choice]]
    else: 
        new_State = lastStateRecord.copy()
        ## Find the dummy session and add it to the session list in the stateRecords 
        for session in States["hour{}-{}".format(arrHour, powerLevel)][choice]['sessions']:
            if session['dcosId'] == -9999:
                new_State[0]['sessions'].append(session)

    # print("New state record TS:",pd.Timestamp(new_State[0]['timestamp'],unit='s'))
    # new_State[0]['timestamp'] = unixTime(sessionStart)
    # print("Changed to:",pd.Timestamp(new_State[0]['timestamp'],unit='s'))

    # new_State = json.loads(new_State)

    # TODO: EXPECTED DEMAND IS ALSO INDEXED BY TIMESTAMP, 
    # SO THE TIMESTAMPS SHOULD MATCH TO NOT DROP
    for session in new_State[0]['sessions']:
        if session['dcosId'] == -9999:
            print("Changed dummy to {}".format(sessionId))
            session['dcosId'] = int(sessionId)
        # TODO: Check if the TIMESTAMPS are correct
    
    for session in new_State[0]['sessions']:
        if session['deadline'] <= unixTime(sessionStart):
            print(pd.to_datetime(session['deadline'],unit='s')," <= ",sessionStart)
            
            new_State[0]['sessions'].remove(session)
            print("removed session:",session['dcosId']) ## Check if it is removing correct or wrong session!!

    print("On-going sessions:", len(new_State[0]['sessions']))
    print("New Choice:",choice)

    return new_State 

# TODO
def queryOptimizedStates():
    
    """ 
    Get the optimzied states / power profile informatiom from the server

    ### After we query, convert the optPower list of lists to array 

    """
    
    # optimizedStates = pd.read_pickle(path + "optimizedStates")

    
    table = dynamodb.Table('optimizedStates')
    optimizedStates = table_to_optimizedStates(table)
    return optimizedStates

# TODO
def queryLastStateRecord():
    
    """ 
    query the last realized stateRecords, no new connection since then
    check the current time of the
    Returns: List with 1 element which is the stateRecord
    """

    # TODO: how to keep track of the lastOptTime

    table = dynamodb.Table('stateRecords')
    stateRecord = [table_to_stateRecord(table)]
    # for session in stateRecord[0]['sessions']:
    #     if session['dcosId'] == -9999:
    #         stateRecord[0]['sessions'].remove(session)
    #         print("Dummy shouldnt be in the stateRecords, removed")

    return stateRecord

def dropFinishedSessions(stateRecord,optTime):
    for session in stateRecord[0]['sessions']:
        if session['deadline'] <= unixTime(optTime):
            print(session['deadline'], unixTime(optTime))
            print(pd.to_datetime(session['deadline'],unit='s')," <= ",optTime)
            print("removed session:",session['dcosId']) ## Check if it is removing correct or wrong session!!
            stateRecord[0]['sessions'].remove(session)
    return stateRecord


def postOptimizedStates(optimizedStates, backup = False):

    """ 
    We don't need back up states, we can overwrite this
    OptimizedPrices_v3 also stores the States from that optimization 
    """

    for key_1 in dict(optimizedStates).keys():
        for session_idx in range(len(optimizedStates[key_1]['SCH']['sessions'])):
            optimizedStates[key_1]['SCH']['sessions'][session_idx]['optPower'] = optimizedStates[key_1]['SCH']['sessions'][session_idx]['optPower'].tolist()
        for session_idx in range(len(optimizedStates[key_1]['REG']['sessions'])):
            optimizedStates[key_1]['REG']['sessions'][session_idx]['optPower'] = optimizedStates[key_1]['REG']['sessions'][session_idx]['optPower'].tolist()

    optimizedStates = convert_floats_to_decimals(dict(optimizedStates))

    
    table = dynamodb.Table('optimizedStates')
    # Scan the table

    # ################# First we empty the table #########################
    response = table.scan()

    # Loop through the items and delete each one
    for item in response['Items']:
        table.delete_item(
            Key={
                'arrHour_highPower': item['arrHour_highPower'],
                'choice': item['choice']
            }
        )

    # Populate DynamoDB table
    for hour_highPower, choices in optimizedStates.items():
        for choice, attributes in choices.items():
            table.put_item(
                Item={
                    'arrHour_highPower': hour_highPower,
                    'choice': choice,
                    'monthlyPeak': Decimal(str(attributes['monthlyPeak'])),
                    'timestamp': Decimal(attributes['timestamp']),
                    'sessions': attributes['sessions'],
                    'reg_centsPerHr':attributes['reg_centsPerHr'],
                    'sch_centsPerHr':attributes['sch_centsPerHr'],
                }
            )


    if backup:

        table = dynamodb.Table('optimizedStates_Backup')

        ## Will overwrite the existing items
        # Populate DynamoDB table
        for hour_highPower, choices in optimizedStates.items():
            for choice, attributes in choices.items():
                table.put_item(
                    Item={
                        'arrHour_highPower': hour_highPower,
                        'choice': choice,
                        'monthlyPeak': Decimal(str(attributes['monthlyPeak'])),
                        'timestamp': attributes['timestamp'],
                        'sessions': attributes['sessions'],
                        'reg_centsPerHr':attributes['reg_centsPerHr'],
                        'sch_centsPerHr':attributes['sch_centsPerHr'],
                    }
                )

    return None


# # TODO
def postOptimizedPrices_v3(OptimizedPrices_v3, optTime, backup=False):
    path = "/Users/aysetugbaozturk/Documents/eCal/SlrpEV/SlrpEV_algorithms/station_level_optimizer/Results/"
    
    if 'arrHour' not in OptimizedPrices_v3.columns:
        OptimizedPrices_v3.reset_index(drop=False,inplace=True)

    if not backup:
        """ How many copies can we keep """
        table = dynamodb.Table('optimizedPrices_v3')
        # Scan the table

        # ################# First we empty the table #########################
        response = table.scan()
        # Loop through the items and delete each one
        for item in response['Items']:
            table.delete_item(
                Key={
                    'highPower': item['highPower'],
                    'arrHour': item['arrHour']
                }
            )
        for index, row in OptimizedPrices_v3.dropna().iterrows():
            item = convert_floats_to_decimals(row.to_dict())
            table.put_item(Item=item)

    if backup:
        table = dynamodb.Table('optimizedPrices_v3_Backup')
        for index, row in OptimizedPrices_v3.dropna().iterrows():
            item = convert_floats_to_decimals(row.to_dict())
            table.put_item(Item=item)

    # 2. Keep backup files OptimizedPrices_v3, indexed by the timestamp 

    table = dynamodb.Table('optimizedPrices_v3_History')
    value = stringify_keys(convert_floats_to_decimals(OptimizedPrices_v3.dropna().to_dict()))

    # First, read the latest sort key for the given timestamp.
    response = table.query(
            KeyConditionExpression="#ts = :timestamp_value",
            ExpressionAttributeNames={
                "#ts": "timestamp"
            },
            ExpressionAttributeValues={
                ":timestamp_value": int(unixTime(optTime))
            },
            ScanIndexForward=False,  # Sort in descending order
            Limit=1  # Only need the last (highest sort key) item
        )
    # Determine next sort key
    next_sort_key = 0
    if 'Items' in response and len(response['Items']) > 0:
        next_sort_key = int(response['Items'][0]['table_no']) + 1

    item = {'timestamp':int(unixTime(optTime)),
            'table_no':next_sort_key,
            'optimizedPrices_v3_table': value
            }
    table.put_item(Item=item)

    return None

import pickle

def postStateRecords(stateRecords):
    """ each time there is a new state, post the station state """
        #### State Record History
    
    stateRecord = convert_floats_to_decimals(stateRecords[0])
    if "timestamp" not in stateRecord.keys():
        stateRecord['timestamp'] = int(stateRecord['timeStamp'])
        del stateRecord['timeStamp']

    table = dynamodb.Table('stateRecords_History')
    # First, read the latest sort key for the given timestamp.
    response = table.query(
            KeyConditionExpression="#ts = :timestamp_value",
            ExpressionAttributeNames={
                "#ts": "timestamp"
            },
            ExpressionAttributeValues={
                ":timestamp_value": int(unixTime(stateRecord['timestamp']))
            },
            ScanIndexForward=False,  # Sort in descending order
            Limit=1  # Only need the last (highest sort key) item
        )

    # Determine next sort key
    next_sort_key = 0
    if 'Items' in response and len(response['Items']) > 0:
        print("There are values with this same TS")
        next_sort_key = int(response['Items'][0]['sort_key']) + 1
    print("sk",next_sort_key)
    stateRecord['sort_key'] = int(next_sort_key)

    table = dynamodb.Table('stateRecords_History')
    response = table.put_item(Item=stateRecord)

 

    table = dynamodb.Table('stateRecords')
    response = table.scan()

    # Loop through the items and delete each one
    for item in response['Items']:
        table.delete_item(
            Key={
                'timestamp': item['timestamp'],
                'sort_key': item['sort_key']
            }
        )
    response = table.put_item(Item=stateRecord)

    return None


# # TODO
def queryExpectedDemand():
    """ dataframe, or dictionary 
        indexed by arrHour and powerLevel
        same as V2 
    """

    table = dynamodb.Table('expectedDemand')

    # Fetch all items from DynamoDB table
    response = table.scan()
    items = response['Items']

    # Prepare empty dictionary for DataFrame
    df_dict = {
        'highPower': [],
        'arrHour': [],
        'estDurationHrs': [],
        'estEnergykWh': [],
        'userPower_kW': []
        # Add other columns if they exist
    }

    # Populate the dictionary
    for item in items:
        df_dict['highPower'].append(int(item['highPower']))
        df_dict['arrHour'].append(int(item['arrHour']))
        
        # Convert Decimal to float before appending
        df_dict['estDurationHrs'].append(float(item.get('estDurationHrs', Decimal(0))))
        df_dict['estEnergykWh'].append(float(item.get('estEnergykWh', Decimal(0))))
        df_dict['userPower_kW'].append(float(item.get('userPower_kW', Decimal(0))))
        # Add other columns if they exist

    # Create DataFrame
    expectedDemand = pd.DataFrame(df_dict)

    return expectedDemand

# TODO 
def readNewUserInfo(engineRecords,  new_session_id, dummy=False):

    ##### DCOSID SHOULD BE AN INPUT 
    """ New user connects and makes a selection, 
    We need the arrival hour, power level and choice of the user 
    new_session_start = "connectTime' in the for the dcosID
    new_session_id = "dcosId"
    new_session_power = "highPower" userPower in the userdictionary 
    arrHour = new_session_start.hour
    new_session_choice = "choice" 
    """

    # new_session_start = "connectTime', int 
    # new_session_id = "dcosId", int 
    # new_session_power = "highPower", decimal
    # arrHour = new_session_start.hour, int
    # new_session_choice = "choice", string

    ################ This part is for demonstration #################### --> sessions2 is processed every 5 min 
    if dummy == True:
        new_session_start = pd.Timestamp('2023-09-16 9:04:27')
        new_session_id = 5646
        new_session_power = 1
        arrHour = new_session_start.hour
        new_session_choice = 'SCH'
    ################ This part is for demonstration #################### 
    else: 
        session_data = [(session['session_connectionTime'],
                         session['session_choice'],
                         session['user_id']) for session in engineRecords[0]['sessions'] if session['session_dcosId'] == new_session_id]
                         
        new_session_start = pd.Timestamp(float(session_data[0][0]),unit='s') - pd.Timedelta(hours=7)
       
        arrHour = new_session_start.hour

        choice_mapping = {
        'SCHEDULED': 'SCH',
        'REGULAR': 'REG'
        }

        new_session_choice = choice_mapping.get(session_data[0][1], "REG")
        ### Why this would fail? ### ASK WHY THERE IS TRY - EXCEPT AROUND THIS? 
        # try:
            # Perform query of max_ac_power from Users table
        table = dynamodb.Table("Users")
        response = table.query(
            KeyConditionExpression = Key('user_id').eq(session_data[0][2]))
        print("User response length:", len(response['Items']))
        try: 
            max_chg_power = min(response['Items'][0]['vehicle_chargeRate'], response['Items'][0]['max_ac_power'])
            print(max_chg_power)
        except:
            # print(response['Items'][0])
            try: 
                max_chg_power = response['Items'][0]['vehicle_chargeRate']
                
            except: 
                print("Don't know user power for userId:",session_data[0][2])
                max_chg_power = 3300
            
        # except:
        #     ### INITIALLY THIS WAS COMING FROM THE priceInfo.vehicle_chargeRate
        #     max_chg_power = response['Items'][0]['vehicle_chargeRate']
     
        # check charging power of user
        if(max_chg_power < 5000):
            new_session_power = 0
        else:
            new_session_power = 1

    return  new_session_start, arrHour, new_session_choice, new_session_id, new_session_power



# def triggerOptimizationAfterNewConnection():

#     """ Event based optimization """

#     new_session_start, arrHour, new_session_choice, new_session_id, new_session_power = readNewUserInfo()
#     States = queryOptimizedStates()
#     new_State = updateStateRecord(States, arrHour, new_session_choice, new_session_power, new_session_id, new_session_start) # This wont work

#     ### New State 
#     stateRecords = new_State

#     expectedDemand = queryExpectedDemand()

#     ### New session start will be the new Optimization Time
#     optTime = new_session_start

#     # unixOptTime = unixTime(optTime)
#     # print(unixOptTime)
#     arrival_hour = optTime.hour

#     OptimizedStates, optimizedPrices_v3  = generateOptPricePower(stateRecords, 
#                                                         expectedDemand,
#                                                         optTime, 
#                                                         arrival_hour,TOU_tariff=TOU_15min())     ##### POTENTIAL FUTURE STATES 
    
#     postOptimizedStates(OptimizedStates)
#     postOptimizedPrices_v3(optimizedPrices_v3, optTime)
#     postStateRecords(stateRecords)

def triggerOptimizationAfterNewConnection_v2(engineRecords):

    print("Sessions in engineRecords:",len(engineRecords[0]['sessions']))

    if len(engineRecords[0]['sessions']) > 0:
        
        stateRecord = queryLastStateRecord()
        #### One source of error could be, the deadline we predicted can be shorter than the actual stay of the vehicle.
        #### Then we drop that session from the stateRecords, but when we compare with the engineRecords its still in the records. 
        #### Then it will appear in the new_connections list. 
        lastOptTime = stateRecord[0]['timestamp']
        # lastOptTime = 1694555105
        print("Last OPT:", pd.Timestamp(lastOptTime,unit='s'))
        # if len(stateRecord[0]['sessions'] ) > 0:
        #     state_dcosIds = {session['dcosId'] for session in stateRecord[0]['sessions']} # This could be empty 
        

        new_connections = 0
        new_opt_time = []
        States=queryOptimizedStates()
        on_going_sessions = []
        for session in engineRecords[0]['sessions']:
            # if session['dcosId'] not in state_dcosIds:
            # print("Session connection time:", pd.Timestamp(session['session_connectionTime'],unit='s'))
            session_connection_CA_Time = session['session_connectionTime'] - 3600 * 8
            print("Session connection local time:", pd.Timestamp(session_connection_CA_Time,unit='s'))
            on_going_sessions.append(session['session_dcosId'])

            if session_connection_CA_Time > lastOptTime:
                new_session_id = session['session_dcosId']
                new_session_start, arrHour, new_session_choice, new_session_id, new_session_power = readNewUserInfo(engineRecords,new_session_id,dummy = False)
                new_opt_time.append(new_session_start)
                print("New session start local time:", pd.Timestamp(new_session_start,unit='s'))
                new_connections += 1
                ## Check this part carefully 
                if new_session_id not in [session['dcosId'] for session in stateRecord[0]['sessions']]:
                    new_State = updateStateRecord(States, stateRecord, arrHour, new_session_choice, new_session_power, new_session_id, new_session_start, connection_count_helper = new_connections)
                    stateRecord = new_State
                else:
                    print("Duplicated session detected")
        
        # Extract the sessions from the first record
        sessions = stateRecord[0]['sessions']
        session_cnt_from_last_connection = len(sessions)
        
        # Filter sessions: keep only those whose dcosId is in on_going_sessions
        filtered_sessions = [session for session in sessions if session['dcosId'] in on_going_sessions]
        
        # Update the stateRecord with the filtered sessions
        stateRecord[0]['sessions'] = filtered_sessions
        if len(filtered_sessions) < session_cnt_from_last_connection:
            print("Dropped disconnected sessions")



        if new_connections > 0:
            ### New State 
            print("Mew connection count:",new_connections)
            expectedDemand = queryExpectedDemand()
            ### New session start will be the new Optimization Time
            optTime = max(new_opt_time)
            print("New state record TS:",pd.Timestamp(new_State[0]['timestamp'],unit='s'))
            new_State[0]['timestamp'] = unixTime(optTime)  
            print("Changed to:",pd.Timestamp(new_State[0]['timestamp'],unit='s'))
            

            # unixOptTime = unixTime(optTime)
            # print(unixOptTime)
            arrival_hour = optTime.hour

            OptimizedStates, optimizedPrices_v3  = generateOptPricePower(stateRecord, 
                                                                expectedDemand,
                                                                optTime,
                                                                arrival_hour,
                                                                opt_horizon=6,
                                                                TOU_tariff=TOU_15min())     ##### POTENTIAL FUTURE STATES 
            
            postOptimizedStates(OptimizedStates)
            postOptimizedPrices_v3(optimizedPrices_v3, optTime)

            postStateRecords(stateRecord)
        
        else:
            return None
        






def triggerTimeBasedOptimization():

    """ 
    Time based optimization 
    Trigger optimziation after every 4 hours, do this by lambda functions from AWS 
    queryLastStateRecord: We check the last state record, drop the vehicles from the state record if there are no on going sessions after the optTime

    """

    optTime = currentTime15min() ## currentTime rounded down to 15min 
    # optTime = pd.Timestamp(year=2023,month=10, day = 9 , hour = 11 , minute= 0)
    optHour = optTime.hour
    stateRecords = queryLastStateRecord() # List with one element #TODO: Fix this, why its a list? 
    stateRecords = dropFinishedSessions(stateRecords,optTime)
    
    stateRecords[0]['timestamp'] = unixTime(optTime)
    print(stateRecords[0]['timestamp'])
    expectedDemand = queryExpectedDemand()

    OptimizedStates, optimizedPrices_v3  = generateOptPricePower(stateRecords, 
                                                        expectedDemand,
                                                        optTime, 
                                                        optHour,
                                                        opt_horizon=5, TOU_tariff=TOU_15min())   ##### POTENTIAL FUTURE STATES 
    
    postOptimizedStates(OptimizedStates)
    postOptimizedPrices_v3(optimizedPrices_v3, optTime)
    postStateRecords(stateRecords)


def triggerBackUpOptimization():

    """ Generates the 24hr states and optimal solution for NO VEHICLE Case.
        Think about the very first user. 
        Also call this function every-day to generate new backup solutions 
    """

    # Think about the very first user, everything should be ready for that user

    ##### IF we trigger Opt before the new day, then there will be pro

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
    stateRecords = queryLastStateRecord() # List with one element #TODO: Fix this, why its a list? 
    emptyState = emptyStateRecord(newOptDate)
    emptyState[0]['monthlyPeak'] = stateRecords[0]['monthlyPeak']
    print("Monthly Peak:", emptyState[0]['monthlyPeak'])
    # emptyState[0]['monthlyPeak'] = 0
    expectedDemand = queryExpectedDemand()
    print("Monthly Peak:", emptyState[0]['monthlyPeak'])

    optimizedStates, optimizedPrices_v3 = generateOptPricePower(emptyState, expectedDemand, newOptDate, 0 , opt_horizon=24, TOU_tariff=TOU_15min())
    
    postOptimizedStates(optimizedStates, backup=True)
    postOptimizedPrices_v3(optimizedPrices_v3, newOptDate, backup=True)
    postStateRecords(emptyState)

def main():
    # Every 24 hours 
    # print("Backup Opt. Running")
    # triggerBackUpOptimization()

    # # print("Event Opt. Running")
    # # triggerOptimizationAfterNewConnection()

    # # # Initialize DynamoDB resource and table
    # table = dynamodb.Table('EngineRecords')  # Replace with your table name

    # # Your known recordID
    # record_id_to_query = 'd4db8d8a78b87e88fac32696b72a1425bc613d57b6d7cd04c164d255df97a512'

    # # Query the table
    # response = table.query(
    #     KeyConditionExpression=boto3.dynamodb.conditions.Key('recordID').eq(record_id_to_query)
    # )

    # # Access items
    # engineRecords = response['Items']
    
    # triggerOptimizationAfterNewConnection_v2(engineRecords)

    # Every 4 hours 
    print("Timed Opt. Running")
    triggerTimeBasedOptimization()

    # postStateRecords(stateRecords)

    return None


if __name__ == "__main__":
    main()
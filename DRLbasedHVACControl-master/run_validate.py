import os.path
import sys
import shutil
import datetime
import time

import random
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('agg')
# matplotlib.use('TkAgg')
import matplotlib.animation as animation
import numpy as np
import torch.cuda
from tqdm import tqdm
import copy

from osm2idf import osm2idf
# from callback_function import callback_function
from data_center import Data_Center
from plot import Drawing
from agent import DQN
from tools import HVAC_setting_value, ReplayBuffer, save_to_csv

# Eplus_Dir = "E:\eplus"
# sys.path.insert(0, Eplus_Dir)
import pyenergyplus
from pyenergyplus.api import EnergyPlusAPI


def update_plot(draw):
    is_x_start = False
    is_x_end = False
    for i in DATA.x:
        if i.hour == 0 and i.minute == 0:
            draw.ax.axvline(i, linewidth=10, color='#ebfced', alpha=0.7)
            continue

        # if i.hour == 8 and i.minute == 0:
        #     x_start = i
        #     is_x_start = True
        # if i.hour == 21 and i.minute == 0:
        #     x_end = i
        #     is_x_end = True
        # if is_x_end and is_x_start:
        #     draw.ax.axvspan(x_start, x_end, alpha=0.7, color='#ebfced')

    draw.set_ax_view()
    # draw.ax.set_xlim(DATA.x[-432], DATA.x[-1])
    # draw.ax.set_ylim(-25, 40)
    # line1, = ax.plot(DATA.x, DATA.Zone_Air_Relative_Humidity_1, label="Zone Temperature")

    # draw.ax.plot(DATA.x, DATA.Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass,
    #      label="CO2 mass", color='black', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Site_Outdoor_Air_Drybulb_Temperature,
                 label="Outdoor Temperature", color='#FFD700', linewidth=1)

    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_1,
                 label="Zone 1 Temperature", color='#48D1CC', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_2,
                 label="Zone 2 Temperature", color='#7FFFAA', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_3,
                 label="Zone 3 Temperature", color='#7B68EE', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_4,
                 label="Zone 4 Temperature", color='#FFD700', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_5,
                 label="Zone 5 Temperature", color='#20B2AA', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Air_Temperature_6,
                 label="Zone 6 Temperature", color='#FF4500', linewidth=1)
    draw.ax.plot(DATA.x, DATA.Zone_Mean_Temperature,
                 label="Zone Mean Temperature", color='#20B2BB', linewidth=5, alpha=0.5)


    # draw.ax.plot(DATA.x, DATA.Zone_Thermostat_Heating_Setpoint_Temperature_6,
    #              label="DATA.Zone_Thermostat_Heating_Setpoint_Temperature_1", color='red', linewidth=0.5,
    #              linestyle='-.')
    # draw.ax.plot(DATA.x, DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_6,
    #              label="DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_1", color='cyan', linewidth=0.5,
    #              linestyle='-.')

    if DATA.train_switch:
        scaled_reward = [x * 0.05 if x is not None else None for x in DATA.reward]
        draw.ax.plot(DATA.x, scaled_reward,
                     label="Reward", color='grey', linewidth=1)



    draw.ax2.plot(DATA.x, DATA.Electricity_HVAC,
                  label="Electricity_HVAC", color='red', linewidth=0.5)

    if draw.is_ion:
        plt.pause(1)
    else:
        plt.show()
    # fig.canvas.draw()
    # plt.show(block=True)


def callback_function(EPstate):
    api = EnergyPlusAPI()
    if not DATA.is_handle:
        if not api.exchange.api_data_fully_ready(EPstate):
            # print(('\033[33mStill waiting for api\033[0m'))
            return
        else:
            DATA.is_handle = True
            # print('\033[32mApi for data exchange is fully ready\033[0m')

            '''Define variable handles'''
            DATA.handle_Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass = api.exchange.get_variable_handle(EPstate, 'Environmental Impact Total CO2 Emissions Carbon Equivalent Mass', 'Thermal Zone 1')
            # This is to get handles for ENVIRONMENT
            DATA.handle_Site_Outdoor_Air_Drybulb_Temperature = api.exchange.get_variable_handle(EPstate, u"Site Outdoor Air Drybulb Temperature", u"ENVIRONMENT")
            DATA.handle_Site_Wind_Speed                      = api.exchange.get_variable_handle(EPstate, u"Site Wind Speed", u"ENVIRONMENT")
            DATA.handle_Site_Wind_Direction                  = api.exchange.get_variable_handle(EPstate, u"Site Wind Direction", u"ENVIRONMENT")
            DATA.handle_Site_Solar_Azimuth_Angle             = api.exchange.get_variable_handle(EPstate, u"Site Solar Azimuth Angle", u"ENVIRONMENT")
            DATA.handle_Site_Solar_Altitude_Angle            = api.exchange.get_variable_handle(EPstate, u"Site Solar Altitude Angle", u"ENVIRONMENT")
            DATA.handle_Site_Solar_Hour_Angle                = api.exchange.get_variable_handle(EPstate, u"Site Solar Hour Angle", u"ENVIRONMENT")

            # This is to get handles for Thermal Zone 1
            DATA.handle_Zone_Air_Relative_Humidity_1                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 1')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_1          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 1')
            DATA.handle_Zone_Infiltration_Mass_1                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 1')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_1             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 1')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 1')
            DATA.handle_Zone_Air_Temperature_1                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 1')
            DATA.handle_Zone_Mean_Radiant_Temperature_1                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 1')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_1 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 1')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_1 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 1')

            # This is to get handles for Thermal Zone 2
            DATA.handle_Zone_Air_Relative_Humidity_2                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 2')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_2          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 2')
            DATA.handle_Zone_Infiltration_Mass_2                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 2')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_2             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 2')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_2   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 2')
            DATA.handle_Zone_Air_Temperature_2                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 2')
            DATA.handle_Zone_Mean_Radiant_Temperature_2                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 2')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_2 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 2')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_2 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 2')

            # This is to get handles for Thermal Zone 3
            DATA.handle_Zone_Air_Relative_Humidity_3                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 3')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_3          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 3')
            DATA.handle_Zone_Infiltration_Mass_3                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 3')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_3             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 3')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_3   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 3')
            DATA.handle_Zone_Air_Temperature_3                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 3')
            DATA.handle_Zone_Mean_Radiant_Temperature_3                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 3')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_3 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 3')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_3 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 3')

            # This is to get handles for Thermal Zone 4
            DATA.handle_Zone_Air_Relative_Humidity_4                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 4')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_4          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 4')
            DATA.handle_Zone_Infiltration_Mass_4                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 4')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_4             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 4')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_4   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 4')
            DATA.handle_Zone_Air_Temperature_4                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 4')
            DATA.handle_Zone_Mean_Radiant_Temperature_4                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 4')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_4 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 4')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_4 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 4')

            # This is to get handles for Thermal Zone 5
            DATA.handle_Zone_Air_Relative_Humidity_5                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 5')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_5          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 5')
            DATA.handle_Zone_Infiltration_Mass_5                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 5')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_5             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 5')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_5   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 5')
            DATA.handle_Zone_Air_Temperature_5                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 5')
            DATA.handle_Zone_Mean_Radiant_Temperature_5                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 5')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_5 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 5')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_5 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 5')

            # This is to get handles for Thermal Zone 6
            DATA.handle_Zone_Air_Relative_Humidity_6                   = api.exchange.get_variable_handle(EPstate, 'Zone Air Relative Humidity', 'Thermal Zone 6')
            DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_6          = api.exchange.get_variable_handle(EPstate, 'Zone Windows Total Heat Gain Energy', 'Thermal Zone 6')
            DATA.handle_Zone_Infiltration_Mass_6                       = api.exchange.get_variable_handle(EPstate, 'Zone Infiltration Mass', 'Thermal Zone 6')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_6             = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass', 'Thermal Zone 6')
            DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_6   = api.exchange.get_variable_handle(EPstate, 'Zone Mechanical Ventilation Mass Flow Rate', 'Thermal Zone 6')
            DATA.handle_Zone_Air_Temperature_6                         = api.exchange.get_variable_handle(EPstate, 'Zone Air Temperature', 'Thermal Zone 6')
            DATA.handle_Zone_Mean_Radiant_Temperature_6                = api.exchange.get_variable_handle(EPstate, 'Zone Mean Radiant Temperature', 'Thermal Zone 6')
            DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_6 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Heating Setpoint Temperature', 'Thermal Zone 6')
            DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_6 = api.exchange.get_variable_handle(EPstate, 'Zone Thermostat Cooling Setpoint Temperature', 'Thermal Zone 6')

            # This is to get METER handles for energy consumption
            DATA.handle_Electricity_Facility = api.exchange.get_meter_handle(EPstate, 'Electricity:Facility')
            DATA.handle_Electricity_HVAC     = api.exchange.get_meter_handle(EPstate, 'Electricity:HVAC')
            DATA.handle_Heating_Electricity  = api.exchange.get_meter_handle(EPstate, 'Heating:Electricity')
            DATA.handle_Cooling_Electricity  = api.exchange.get_meter_handle(EPstate, 'Cooling:Electricity')




            '''actuator'''
            # This is to get handles for actuators for each zone
            DATA.handle_Heating_Setpoint_1 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control',  'Heating Setpoint', 'Thermal Zone 1')
            DATA.handle_Cooling_Setpoint_1 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 1')

            DATA.handle_Heating_Setpoint_2 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 2')
            DATA.handle_Cooling_Setpoint_2 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 2')

            DATA.handle_Heating_Setpoint_3 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 3')
            DATA.handle_Cooling_Setpoint_3 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 3')

            DATA.handle_Heating_Setpoint_4 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 4')
            DATA.handle_Cooling_Setpoint_4 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 4')

            DATA.handle_Heating_Setpoint_5 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 5')
            DATA.handle_Cooling_Setpoint_5 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 5')

            DATA.handle_Heating_Setpoint_6 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 6')
            DATA.handle_Cooling_Setpoint_6 = api.exchange.get_actuator_handle(EPstate, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 6')

            # print(DATA.get_handles_state())
            if not DATA.get_handles_state():
                print('\033[31mInvalid handles, check spelling and sensor/actuator availability\033[0m')
                sys.exit(1)
    if api.exchange.warmup_flag(EPstate):
        return

    # print(f'区域1温度为：{api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_1)}')
    '''
    Retrieve data using variable handles 
    '''
    DATA.Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass.append(api.exchange.get_variable_value(EPstate, DATA.handle_Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass))
    # ENVIRONMENT
    DATA.Site_Outdoor_Air_Drybulb_Temperature.append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Outdoor_Air_Drybulb_Temperature))
    DATA.Site_Wind_Speed                     .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Wind_Speed))
    DATA.Site_Wind_Direction                 .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Wind_Direction))
    DATA.Site_Solar_Azimuth_Angle            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Solar_Azimuth_Angle))
    DATA.Site_Solar_Altitude_Angle           .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Solar_Altitude_Angle))
    DATA.Site_Solar_Hour_Angle               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Site_Solar_Hour_Angle))

    # Thermal Zone 1
    DATA.Zone_Air_Relative_Humidity_1                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_1))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_1         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_1))
    DATA.Zone_Infiltration_Mass_1                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_1))
    DATA.Zone_Mechanical_Ventilation_Mass_1            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_1))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_1  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1))
    DATA.Zone_Air_Temperature_1                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_1))
    DATA.Zone_Mean_Radiant_Temperature_1               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_1))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_1.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_1))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_1.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_1))


    # Thermal Zone 2
    DATA.Zone_Air_Relative_Humidity_2                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_2))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_2         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_2))
    DATA.Zone_Infiltration_Mass_2                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_2))
    DATA.Zone_Mechanical_Ventilation_Mass_2            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_2))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_2  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_2))
    DATA.Zone_Air_Temperature_2                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_2))
    DATA.Zone_Mean_Radiant_Temperature_2               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_2))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_2.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_2))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_2.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_2))

    # Thermal Zone 3
    DATA.Zone_Air_Relative_Humidity_3                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_3))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_3         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_3))
    DATA.Zone_Infiltration_Mass_3                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_3))
    DATA.Zone_Mechanical_Ventilation_Mass_3            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_3))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_3  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_3))
    DATA.Zone_Air_Temperature_3                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_3))
    DATA.Zone_Mean_Radiant_Temperature_3               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_3))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_3.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_3))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_3.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_3))

    # Thermal Zone 4
    DATA.Zone_Air_Relative_Humidity_4                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_4))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_4         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_4))
    DATA.Zone_Infiltration_Mass_4                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_4))
    DATA.Zone_Mechanical_Ventilation_Mass_4            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_4))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_4  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_4))
    DATA.Zone_Air_Temperature_4                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_4))
    DATA.Zone_Mean_Radiant_Temperature_4               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_4))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_4.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_4))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_4.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_4))

    # Thermal Zone 5
    DATA.Zone_Air_Relative_Humidity_5                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_5))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_5         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_5))
    DATA.Zone_Infiltration_Mass_5                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_5))
    DATA.Zone_Mechanical_Ventilation_Mass_5            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_5))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_5  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_5))
    DATA.Zone_Air_Temperature_5                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_5))
    DATA.Zone_Mean_Radiant_Temperature_5               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_5))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_5.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_5))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_5.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_5))

    # Thermal Zone 6
    DATA.Zone_Air_Relative_Humidity_6                  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Relative_Humidity_6))
    DATA.Zone_Windows_Total_Heat_Gain_Energy_6         .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Windows_Total_Heat_Gain_Energy_6))
    DATA.Zone_Infiltration_Mass_6                      .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Infiltration_Mass_6))
    DATA.Zone_Mechanical_Ventilation_Mass_6            .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_6))
    DATA.Zone_Mechanical_Ventilation_Mass_Flow_Rate_6  .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_6))
    DATA.Zone_Air_Temperature_6                        .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_6))
    DATA.Zone_Mean_Radiant_Temperature_6               .append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Mean_Radiant_Temperature_6))
    DATA.Zone_Thermostat_Heating_Setpoint_Temperature_6.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Heating_Setpoint_Temperature_6))
    DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_6.append(api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_6))

    DATA.Zone_Mean_Temperature.append((DATA.Zone_Air_Temperature_1[-1] +
                                      DATA.Zone_Air_Temperature_2[-1] +
                                      DATA.Zone_Air_Temperature_3[-1] +
                                      DATA.Zone_Air_Temperature_4[-1] +
                                      DATA.Zone_Air_Temperature_5[-1] +
                                      DATA.Zone_Air_Temperature_6[-1])/6)

    # Energy
    DATA.Electricity_Facility.append(api.exchange.get_meter_value(EPstate, DATA.handle_Electricity_Facility))
    DATA.Electricity_HVAC    .append(api.exchange.get_meter_value(EPstate, DATA.handle_Electricity_HVAC))
    DATA.Heating_Electricity .append(api.exchange.get_meter_value(EPstate, DATA.handle_Heating_Electricity))
    DATA.Cooling_Electricity .append(api.exchange.get_meter_value(EPstate, DATA.handle_Cooling_Electricity))


    # Time
    # T_year = api.exchange.year(EPstate)
    T_year             = 2023
    T_month            = api.exchange.month(EPstate)
    T_day              = api.exchange.day_of_month(EPstate)
    T_hour             = api.exchange.hour(EPstate)
    T_minute           = api.exchange.minutes(EPstate)
    T_current_time     = api.exchange.current_time(EPstate)
    T_actual_date_time = api.exchange.actual_date_time(EPstate)
    T_actual_time      = api.exchange.actual_time(EPstate)
    T_time_step        = api.exchange.zone_time_step_number(EPstate)

    DATA.T_years            .append(T_year)
    DATA.T_months           .append(T_month)
    DATA.T_days             .append(T_day)
    DATA.T_hours            .append(T_hour)
    DATA.T_minutes          .append(T_minute)
    DATA.T_current_times    .append(T_current_time)
    DATA.T_actual_date_times.append(T_actual_date_time)
    DATA.T_actual_times     .append(T_actual_time)
    DATA.T_time_steps       .append(T_time_step)

    timedelta = datetime.timedelta()
    if T_minute >= 60:
        T_minute = 59
        timedelta += datetime.timedelta(minutes=1)

    dt = datetime.datetime(
        year=T_year,
        month=T_month,
        day=T_day,
        hour=T_hour,
        minute=T_minute
    )
    dt += timedelta
    DATA.x.append(dt)


    # if T_hour == 8:
    #     api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, 28)
    #     print(f'在调整setpoint以后区域1温度为：{api.exchange.get_variable_value(EPstate, DATA.handle_Zone_Air_Temperature_1)}')
    # if T_hour == 12:
    #     api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, 24)
    #
    # if T_hour == 16:
    #     api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, 26)
    #
    # # if count == draw.x_view:
    # #     api.exchange.set_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1, 20)
    worktime_indicator = False
    if 8 < T_hour < 21:
        worktime_indicator = True
    else:
        worktime_indicator = False

    if 5 <= T_month <= 10:
        is_summer = 10
    else:
        is_summer = 1


    '''DQN training'''
    if DATA.train_switch and worktime_indicator:
        done = False

        ''' current_state and also the 'next_state' for the last episode'''
        s1 = DATA.Zone_Air_Temperature_1[-1]
        s2 = DATA.Zone_Air_Temperature_2[-1]
        s3 = DATA.Zone_Air_Temperature_3[-1]
        s4 = DATA.Zone_Air_Temperature_4[-1]
        s5 = DATA.Zone_Air_Temperature_5[-1]
        s6 = DATA.Zone_Air_Temperature_6[-1]
        s7 = DATA.Site_Outdoor_Air_Drybulb_Temperature[-1]
        s8 = worktime_indicator
        s9 = api.exchange.get_actuator_value(EPstate, DATA.handle_Heating_Setpoint_1)
        s10 = api.exchange.get_actuator_value(EPstate, DATA.handle_Heating_Setpoint_2)
        s11 = api.exchange.get_actuator_value(EPstate, DATA.handle_Heating_Setpoint_3)
        s12 = api.exchange.get_actuator_value(EPstate, DATA.handle_Heating_Setpoint_4)
        s13 = api.exchange.get_actuator_value(EPstate, DATA.handle_Heating_Setpoint_5)
        s14 = api.exchange.get_actuator_value(EPstate, DATA.handle_Heating_Setpoint_6)
        s15 = api.exchange.get_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_1)
        s16 = api.exchange.get_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_2)
        s17 = api.exchange.get_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_3)
        s18 = api.exchange.get_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_4)
        s19 = api.exchange.get_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_5)
        s20 = api.exchange.get_actuator_value(EPstate, DATA.handle_Cooling_Setpoint_6)
        s21 = is_summer

        E = DATA.Electricity_HVAC[-1]
        # s8 = DATA.T_months[-1]
        # s9 = DATA.T_days[-1]
        # s10 = DATA.T_hours[-1]
        state0 = [s1/100, s2/100, s3/100, s4/100, s5/100, s6/100,
                  s7, s8, ]
                  #s9/100, s10/100, s11/100, s12/100, s13/100, s14/100, s15/100, s16/100, s17/100, s18/100, s19/100, s20/100, s21/100]
        DATA.state.append(state0)
        Temp_list = [s1, s2, s4, s5, s6]
        Temp_mean = sum(Temp_list)/len(Temp_list)

        ''' reward for last episode'''
        #  Energy reward
        if is_summer == 10:
            factor_E = 5e-5
            # factor_E = (5e-4)/(abs(Temp_mean- s7)+1)
            # factor_E = 0
        else:
            factor_E = 5e-6
            # factor_E = 0
        reward_E = - factor_E * E

        #  Temperature reward
        if worktime_indicator:
            positive = 3
            factor_T = 5
        else:
            positive = 0
            factor_T = 0
        reward_T_list = []
        for T in Temp_list:
            if 24 < T < 26:
                delta = abs(T-25)
                if delta < 0.1:
                    reward_T_list.append(50)
                else:
                    reward_T_list.append(positive * (1/delta))
            elif T < 24:
                reward_T_list.append(-abs(T - 24) ** 2 * factor_T)
            else:
                reward_T_list.append(-abs(T - 26) ** 2 * factor_T)
        T_standard_deviation = -np.std(Temp_list, ddof=1)
        reward_T_list.append(T_standard_deviation * 2)
        reward_T = np.sum(reward_T_list)


        #  wear-out reward
        if DATA.wear_out_flag:
            if len(DATA.action) >= 2:
                last_a = DATA.HVAC_action_map[DATA.action[-2]]
                current_a = DATA.HVAC_action_map[DATA.action[-1]]
            elif len(DATA.action) == 1:
                last_a = random.choice(DATA.HVAC_action_map)
                current_a = DATA.HVAC_action_map[DATA.action[-1]]
            else:
                last_a = random.choice(DATA.HVAC_action_map)
                current_a = random.choice(DATA.HVAC_action_map)

            n_shift_signal = sum(1 for x, y in zip(last_a, current_a) if x != y)
            factor_S = 3
            reward_S = - factor_S * n_shift_signal
            # if DATA.wear_out_flag:
            reward_ = reward_E + reward_T + reward_S
        else:
            reward_S = False
            reward_ = reward_E + reward_T
        DATA.reward.append(reward_)
        DATA.reward_random_memory.append(reward_)

        '''Temperature Violation'''
        Temp_mean_violation = 0
        if worktime_indicator:
            if 24 <= Temp_mean <= 26:
                Temp_mean_violation = 0
            else:
                Temp_mean_violation = 1
            # elif Temp_mean > 26:
            #     Temp_mean_violation = Temp_mean - 25
            # else:
            #     Temp_mean_violation = 24 - Temp_mean
        DATA.Temp_mean_violation.append(Temp_mean_violation)

        Temp_violation = []
        if worktime_indicator:
            for T in Temp_list:
                if T > 26:
                    Temp_violation.append(T - 26)
                elif T < 24:
                    Temp_violation.append(24 - T)
        Temp_violation = np.sum(Temp_violation)
        DATA.Temp_violation.append(Temp_violation)

        #  take action
        action0 = EPagent.take_action_for_validation(state0)
        DATA.action.append(action0)
        action_list = DATA.HVAC_action_map[action0]
        #  put action into EP for next simulation step
        actuators = [
            DATA.handle_Heating_Setpoint_1,
            DATA.handle_Cooling_Setpoint_1,
            DATA.handle_Heating_Setpoint_2,
            DATA.handle_Cooling_Setpoint_2,
            DATA.handle_Heating_Setpoint_3,
            DATA.handle_Cooling_Setpoint_3,
            DATA.handle_Heating_Setpoint_4,
            DATA.handle_Cooling_Setpoint_4,
            DATA.handle_Heating_Setpoint_5,
            DATA.handle_Cooling_Setpoint_5,
            DATA.handle_Heating_Setpoint_6,
            DATA.handle_Cooling_Setpoint_6,
        ]

        end = 0
        for i in action_list:
            # set_value = HVAC_setting_value(i)
            # print('原来h', api.exchange.get_actuator_value(EPstate, actuators[2 * i_index]))
            # print('原来c', api.exchange.get_actuator_value(EPstate, actuators[2 * i_index + 1]))
            api.exchange.set_actuator_value(EPstate, actuators[2*end], i)
            # print(api.exchange.get_actuator_value(EPstate, actuators[2*i_index]))
            api.exchange.set_actuator_value(EPstate, actuators[2*end+1], i+2)
            # print(api.exchange.get_actuator_value(EPstate, actuators[2*i_index+1]))
            end += 1

        #  Done
        if Temp_violation < 6:
            done = True
        DATA.done.append(done)

    else:
        DATA.reward.append(None)
        DATA.reward_random_memory.append(None)
        DATA.loss.append(None)
        DATA.loss_random_memory.append(None)


    DATA.count += 1
    """Plot"""

    if draw.is_ion:
        if DATA.count > draw.x_view:
            update_plot(draw)

    if DATA.train_switch and worktime_indicator:
        if DATA.count % 2000 == 0:
            print(
                f'Time: {DATA.count} / {dt}   Temp: {Temp_mean:.2f} / 25 / {s7:.2f}   Reward(T/E/S): {reward_T:.2f} / {reward_E:.2f} / {reward_S:.2f}')

    DATA.is_handle = False


class Run_EPlus():

    def __init__(self, weather_Dir, out_Dir, IDF_Dir, weights_Dir=None):
        print(f'Your current "EnergyPlusAPI" version is: V-{pyenergyplus.api.EnergyPlusAPI.api_version()}')
        print(f'Your current "pyenergyplus" directory is: {pyenergyplus.api.api_path()}')

        self.weather_Dir = weather_Dir
        self.out_Dir = out_Dir
        self.IDF_Dir = IDF_Dir
        self.weights_Dir = weights_Dir

        self.isPrecondition1 = True
        self.isPrecondition2 = True
        self.isPrecondition3 = True
        self.isPrecondition4 = True

        if not os.path.exists(self.IDF_Dir):
            self.isPrecondition1 = False
            raise FileNotFoundError(f"The path '{self.IDF_Dir}' does not exist.")
        if not os.path.exists(self.weather_Dir):
            self.isPrecondition2 = False
            raise FileNotFoundError(f"The path '{self.weather_Dir}' does not exist.")
        try:
            if not os.path.exists(self.out_Dir):
                os.makedirs(self.out_Dir)
            else:
                print(f"Directory '{self.out_Dir}' already exists.")
                self.remove_folder(self.out_Dir)
                os.makedirs(self.out_Dir)
                print(f"A new folder '{self.out_Dir}' has been created.")
        except Exception as e:
            self.isPrecondition3 = False
            # 捕获创建目录时可能发生的任何异常
            raise OSError(f"Could not create the directory '{self.out_Dir}'. Reason: {e}")

        # if DATA.train_switch:
        #     try:
        #         if not os.path.exists(self.weights_Dir):
        #             os.makedirs(self.weights_Dir)
        #         else:
        #             print(f"Directory '{self.weights_Dir}' already exists.")
        #             self.remove_folder(self.weights_Dir)
        #             os.makedirs(self.weights_Dir)
        #             print(f"A new folder '{self.weights_Dir}' has been created.")
        #     except Exception as e:
        #         self.isPrecondition4 = False
        #         # 捕获创建目录时可能发生的任何异常
        #         raise OSError(f"Could not create the directory '{self.weights_Dir}'. Reason: {e}")

        if self.isPrecondition1 and self.isPrecondition2 and self.isPrecondition3 and self.isPrecondition4:
            self.deploy_new_EPstate()
        else:
            raise FileNotFoundError("CANNOT deploy new EneryPlus state, "
                                    "please check if all the needed files has been properly placed")

        if self.IDF_Dir[-4:] == '.osm':
            trans_path = osm2idf(self.IDF_Dir)
            self.IDF_Dir = trans_path.idf_file

    def deploy_new_EPstate(self):
        self.EPapi = EnergyPlusAPI()
        print('EnergyPlus state deployed successfully.')

    def start_simulation(self, iscallback=True, isEPtoConsole=False):
        EPapi = EnergyPlusAPI()
        EPstate = EPapi.state_manager.new_state()
        EPapi.runtime.set_console_output_status(EPstate, isEPtoConsole)  # set EP console output status to False
        if iscallback:
            EPapi.runtime.callback_begin_zone_timestep_after_init_heat_balance(EPstate, callback_function)

        EPapi.runtime.run_energyplus(
            EPstate,
            [
                '-w', self.weather_Dir,
                '-d', self.out_Dir,
                self.IDF_Dir
            ]
        )
        EPapi.state_manager.reset_state(EPstate)
        EPapi.state_manager.delete_state(EPstate)
        # if not DATA.train_switch:
        #     EPapi.state_manager.delete_state(EPstate)
        if DATA.train_switch:
            if not draw.is_ion:
                update_plot(draw)


    def remove_folder(self, path):
        shutil.rmtree(path)
        print(f"The directory '{path}' and all its contents have been removed.")


if __name__ == '__main__':
    weather_Dir = "./weather_data/CHN_Beijing.Beijing.545110_CSWD.epw"
    out_Dir = "./out"
    # IDF_Dir = "./building_model/new_ue_room/1-18/1.18.osm"
    IDF_Dir = "./EPmodel/1.19.osm"
    # IDF_Dir = r"C:\Users\Lee\Desktop\1-18\1.18.osm"
    # IDF_Dir = "./sp/sp.osm"
    weights_Dir = "./weights"
    # IDF_Dir = "./building_model/1.5-1-no-site.osm"

    DATA = Data_Center()
    DATA.train_switch = False
    DATA.wear_out_flag = False

    draw = Drawing(DATA, is_ion=False, is_zoom=False)
    run_instance = Run_EPlus(weather_Dir, out_Dir, IDF_Dir)
    run_instance.start_simulation(iscallback=True, isEPtoConsole=True)

    RHVAC_power = DATA.Electricity_HVAC
    RuleBasedEnergyConsumption =sum(DATA.Electricity_HVAC)

    # save_to_csv(DATA)

    '''DQN training'''
    DATA = Data_Center()
    DATA.initialize_handels()
    DATA.initialize_valuse()
    DATA.count = 0
    DATA.train_switch = True
    DATA.wear_out_flag = True
    draw = Drawing(DATA, is_ion=False, is_zoom=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPagent = DQN(
        state_dim=8,
        action_dim=729,
        lr=0.01,
        gamma=0.9,
        epsilon=0,
        device=device,
        update_interval=72
    )
    # EPagent.target_Q_net.load_state_dict(torch.load('./BESTMODEL/episode_52318.pth'))
    EPagent.target_Q_net.load_state_dict(torch.load('./weights/EPagent_1.pth'))
    EPagent.target_Q_net.eval()

    run_instance = Run_EPlus(weather_Dir, out_Dir, IDF_Dir, weights_Dir)


    print('torch.version: ', torch.__version__)
    print('torch.version.cuda: ', torch.version.cuda)
    print('torch.cuda.is_available: ', torch.cuda.is_available())
    print('torch.cuda.device_count: ', torch.cuda.device_count())
    print('torch.cuda.current_device: ', torch.cuda.current_device())
    device_default = torch.cuda.current_device()
    torch.cuda.device(device_default)
    print('torch.cuda.get_device_name: ', torch.cuda.get_device_name(device_default))
    device = torch.device("cuda")
    print('\n')
    print('Training begins')
    print('\n')

    #  Run the training
    run_instance.start_simulation(iscallback=True, isEPtoConsole=False)

    DQNHVAC_power = DATA.Electricity_HVAC
    DQNEnergyConsumption = sum(DATA.Electricity_HVAC)
    plt.figure(figsize=(20, 10))
    plt.plot(range(1,len(RHVAC_power)+1), RHVAC_power, label="Rule Based Power", color='#5b9bd5', linewidth=3)
    plt.plot(range(1,len(DQNHVAC_power)+1), DQNHVAC_power, label="DQN Based Power", color='#ed7d31', linewidth=3)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.title('Annual Energy Consumption', fontsize=22)
    plt.xlabel('Episode', fontsize=22)
    plt.ylabel('Energy Consumption(J)', fontsize=22)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.scatter(range(1,len(DATA.Temp_mean_violation)+1), DATA.Temp_mean_violation, label="Temp mean violation", color='red', linewidth=2)
    plt.show()

    print(f'Rule-based Energy consumption: {RuleBasedEnergyConsumption:.3f}J')
    print(f'DQN Energy consumption: {DQNEnergyConsumption:.3f}J')
    print('\n')
    # print(f"Energy saving ratio: {1-(DQNEnergyConsumption/RuleBasedEnergyConsumption):.3f}")

    print(f"Temperature Violation ratio: {(sum(DATA.Temp_mean_violation) / len(DATA.Temp_mean_violation)):.3f}")
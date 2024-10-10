from tools import HVAC_action_map

#存储数据
class Data_Center():
    def __init__(self, train_switch=False):
        self.HVAC_action_map = HVAC_action_map()
        self.train_switch = train_switch
        self.is_handle = False
        self.count = 0
        self.minimal_episode = 400
        self.wear_out_flag = False
        """
        This part is for ENVIRONMENT actuator handles
        """
        self.handle_Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass = -1

        self.handle_Site_Outdoor_Air_Drybulb_Temperature = -1
        self.handle_Site_Wind_Speed = -1
        self.handle_Site_Wind_Direction = -1
        self.handle_Site_Solar_Azimuth_Angle = -1
        self.handle_Site_Solar_Altitude_Angle = -1
        self.handle_Site_Solar_Hour_Angle = -1

        """
        This part is for ZONE variable handles
        """
        #zone？内的变量被用来存储模拟环境或实际环境的数据
        # thermal zone 1
        self.handle_Zone_Air_Relative_Humidity_1 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_1 = -1
        self.handle_Zone_Infiltration_Mass_1 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_1 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1 = -1
        self.handle_Zone_Air_Temperature_1 = -1
        self.handle_Zone_Mean_Radiant_Temperature_1 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_1 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_1 = -1

        # thermal zone 2
        self.handle_Zone_Air_Relative_Humidity_2 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_2 = -1
        self.handle_Zone_Infiltration_Mass_2 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_2 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1 = -1
        self.handle_Zone_Air_Temperature_2 = -1
        self.handle_Zone_Mean_Radiant_Temperature_2 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_2 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_2 = -1

        # thermal zone 3
        self.handle_Zone_Air_Relative_Humidity_3 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_3 = -1
        self.handle_Zone_Infiltration_Mass_3 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_3 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_3 = -1
        self.handle_Zone_Air_Temperature_3 = -1
        self.handle_Zone_Mean_Radiant_Temperature_3 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_3 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_3 = -1

        # thermal zone 4
        self.handle_Zone_Air_Relative_Humidity_4 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_4 = -1
        self.handle_Zone_Infiltration_Mass_4 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_4 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_4 = -1
        self.handle_Zone_Air_Temperature_4 = -1
        self.handle_Zone_Mean_Radiant_Temperature_4 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_4 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_4 = -1

        # thermal zone 5
        self.handle_Zone_Air_Relative_Humidity_5 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_5 = -1
        self.handle_Zone_Infiltration_Mass_5 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_5 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_5 = -1
        self.handle_Zone_Air_Temperature_5 = -1
        self.handle_Zone_Mean_Radiant_Temperature_5 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_5 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_5 = -1

        # thermal zone 6
        self.handle_Zone_Air_Relative_Humidity_6 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_6 = -1
        self.handle_Zone_Infiltration_Mass_6 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_6 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_6 = -1
        self.handle_Zone_Air_Temperature_6 = -1
        self.handle_Zone_Mean_Radiant_Temperature_6 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_6 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_6 = -1

        # Energy handles --- meters
        self.handle_Electricity_Facility = -1
        self.handle_Electricity_HVAC = -1
        self.handle_Heating_Electricity = -1
        self.handle_Cooling_Electricity = -1

        self.handle_Electricity_Zone_1 = -1

        # Actuator handles
        self.handle_Heating_Setpoint_1 = -1
        self.handle_Cooling_Setpoint_1 = -1
        self.handle_Heating_Setpoint_2 = -1
        self.handle_Cooling_Setpoint_2 = -1
        self.handle_Heating_Setpoint_3 = -1
        self.handle_Cooling_Setpoint_3 = -1
        self.handle_Heating_Setpoint_4 = -1
        self.handle_Cooling_Setpoint_4 = -1
        self.handle_Heating_Setpoint_5 = -1
        self.handle_Cooling_Setpoint_5 = -1
        self.handle_Heating_Setpoint_6 = -1
        self.handle_Cooling_Setpoint_6 = -1

        self.handles = []



        """
        This part is for data value storage
        """
        self.Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass = []
        # Environment
        self.Site_Outdoor_Air_Drybulb_Temperature = []
        self.Site_Wind_Speed = []
        self.Site_Wind_Direction = []
        self.Site_Solar_Azimuth_Angle = []
        self.Site_Solar_Altitude_Angle = []
        self.Site_Solar_Hour_Angle = []

        # thermal zone 1
        self.Zone_Air_Relative_Humidity_1 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_1 = []
        self.Zone_Infiltration_Mass_1 = []
        self.Zone_Mechanical_Ventilation_Mass_1 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_1 = []
        self.Zone_Air_Temperature_1 = []
        self.Zone_Mean_Radiant_Temperature_1 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_1 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_1 = []

        # thermal zone 2
        self.Zone_Air_Relative_Humidity_2 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_2 = []
        self.Zone_Infiltration_Mass_2 = []
        self.Zone_Mechanical_Ventilation_Mass_2 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_2 = []
        self.Zone_Air_Temperature_2 = []
        self.Zone_Mean_Radiant_Temperature_2 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_2 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_2 = []

        # thermal zone 3
        self.Zone_Air_Relative_Humidity_3 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_3 = []
        self.Zone_Infiltration_Mass_3 = []
        self.Zone_Mechanical_Ventilation_Mass_3 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_3 = []
        self.Zone_Air_Temperature_3 = []
        self.Zone_Mean_Radiant_Temperature_3 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_3 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_3 = []

        # thermal zone 4
        self.Zone_Air_Relative_Humidity_4 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_4 = []
        self.Zone_Infiltration_Mass_4 = []
        self.Zone_Mechanical_Ventilation_Mass_4 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_4 = []
        self.Zone_Air_Temperature_4 = []
        self.Zone_Mean_Radiant_Temperature_4 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_4 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_4 = []

        # thermal zone 5
        self.Zone_Air_Relative_Humidity_5 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_5 = []
        self.Zone_Infiltration_Mass_5 = []
        self.Zone_Mechanical_Ventilation_Mass_5 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_5 = []
        self.Zone_Air_Temperature_5 = []
        self.Zone_Mean_Radiant_Temperature_5 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_5 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_5 = []

        # thermal zone 6
        self.Zone_Air_Relative_Humidity_6 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_6 = []
        self.Zone_Infiltration_Mass_6 = []
        self.Zone_Mechanical_Ventilation_Mass_6 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_6 = []
        self.Zone_Air_Temperature_6 = []
        self.Zone_Mean_Radiant_Temperature_6 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_6 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_6 = []

        self.Zone_Mean_Temperature = []

        # Energy
        self.Electricity_Facility = []
        self.Electricity_HVAC = []
        self.Heating_Electricity = []
        self.Cooling_Electricity = []

        self.Electricity_Zone_1 = []
        # Time
        self.x = []

        self.T_years = []
        self.T_months = []
        self.T_days = []
        self.T_hours = []
        self.T_minutes = []
        self.T_current_times = []
        self.T_actual_date_times = []
        self.T_actual_times = []
        self.T_time_steps = []

        self.T_weekday = []
        self.T_isweekday = []
        self.T_isweekend = []
        self.T_work_time = []


        '''for DQN'''  # 未加后缀的变量都是针对于eposide
        self.state = []
        self.action = []
        self.reward = []
        self.reward_random_memory = []
        self.next_state = []
        self.done = []
        self.loss = []
        self.loss_random_memory = []
        self.Temp_mean_violation = []
        self.Temp_violation = []


        self.loss_epoch = []
        self.reward_epoch = []
        self.reward_accumulative = 0

#获得和存储特定环境参数在self.handles列表中
    def get_handles_state(self):
        self.handles = [
            self.handle_Site_Outdoor_Air_Drybulb_Temperature,
            self.handle_Site_Wind_Speed,
            self.handle_Site_Wind_Direction,
            self.handle_Site_Solar_Azimuth_Angle,
            self.handle_Site_Solar_Altitude_Angle,
            self.handle_Site_Solar_Hour_Angle,

            self.handle_Zone_Air_Relative_Humidity_1,
            self.handle_Zone_Windows_Total_Heat_Gain_Energy_1,
            self.handle_Zone_Infiltration_Mass_1,
            self.handle_Zone_Mechanical_Ventilation_Mass_1,
            self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1,
            self.handle_Zone_Air_Temperature_1,
            self.handle_Zone_Mean_Radiant_Temperature_1,
            self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_1,
            self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_1,

            self.handle_Zone_Air_Relative_Humidity_2,
            self.handle_Zone_Windows_Total_Heat_Gain_Energy_2,
            self.handle_Zone_Infiltration_Mass_2,
            self.handle_Zone_Mechanical_Ventilation_Mass_2,
            self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1,
            self.handle_Zone_Air_Temperature_2,
            self.handle_Zone_Mean_Radiant_Temperature_2,
            self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_2,
            self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_2,

            self.handle_Zone_Air_Relative_Humidity_3,
            self.handle_Zone_Windows_Total_Heat_Gain_Energy_3,
            self.handle_Zone_Infiltration_Mass_3,
            self.handle_Zone_Mechanical_Ventilation_Mass_3,
            self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_3,
            self.handle_Zone_Air_Temperature_3,
            self.handle_Zone_Mean_Radiant_Temperature_3,
            self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_3,
            self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_3,

            self.handle_Zone_Air_Relative_Humidity_4,
            self.handle_Zone_Windows_Total_Heat_Gain_Energy_4,
            self.handle_Zone_Infiltration_Mass_4,
            self.handle_Zone_Mechanical_Ventilation_Mass_4,
            self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_4,
            self.handle_Zone_Air_Temperature_4,
            self.handle_Zone_Mean_Radiant_Temperature_4,
            self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_4,
            self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_4,

            self.handle_Zone_Air_Relative_Humidity_5,
            self.handle_Zone_Windows_Total_Heat_Gain_Energy_5,
            self.handle_Zone_Infiltration_Mass_5,
            self.handle_Zone_Mechanical_Ventilation_Mass_5,
            self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_5,
            self.handle_Zone_Air_Temperature_5,
            self.handle_Zone_Mean_Radiant_Temperature_5,
            self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_5,
            self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_5,

            self.handle_Zone_Air_Relative_Humidity_6,
            self.handle_Zone_Windows_Total_Heat_Gain_Energy_6,
            self.handle_Zone_Infiltration_Mass_6,
            self.handle_Zone_Mechanical_Ventilation_Mass_6,
            self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_6,
            self.handle_Zone_Air_Temperature_6,
            self.handle_Zone_Mean_Radiant_Temperature_6,
            self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_6,
            self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_6,

            self.handle_Electricity_Facility,
            self.handle_Electricity_HVAC,
            self.handle_Heating_Electricity,
            self.handle_Cooling_Electricity,

            self.handle_Heating_Setpoint_1,
            self.handle_Cooling_Setpoint_1,
            self.handle_Heating_Setpoint_2,
            self.handle_Cooling_Setpoint_2,
            self.handle_Heating_Setpoint_3,
            self.handle_Cooling_Setpoint_3,
            self.handle_Heating_Setpoint_4,
            self.handle_Cooling_Setpoint_4,
            self.handle_Heating_Setpoint_5,
            self.handle_Cooling_Setpoint_5,
            self.handle_Heating_Setpoint_6,
            self.handle_Cooling_Setpoint_6,
        ]
        if -1 in self.handles:
            return False
        else:
            return True


    def initialize_valuse(self):
        """
        This part is for data value storage
        """
        self.Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass = []
        # Environment
        self.Site_Outdoor_Air_Drybulb_Temperature = []
        self.Site_Wind_Speed = []
        self.Site_Wind_Direction = []
        self.Site_Solar_Azimuth_Angle = []
        self.Site_Solar_Altitude_Angle = []
        self.Site_Solar_Hour_Angle = []

        # thermal zone 1
        self.Zone_Air_Relative_Humidity_1 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_1 = []
        self.Zone_Infiltration_Mass_1 = []
        self.Zone_Mechanical_Ventilation_Mass_1 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_1 = []
        self.Zone_Air_Temperature_1 = []
        self.Zone_Mean_Radiant_Temperature_1 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_1 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_1 = []

        # thermal zone 2
        self.Zone_Air_Relative_Humidity_2 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_2 = []
        self.Zone_Infiltration_Mass_2 = []
        self.Zone_Mechanical_Ventilation_Mass_2 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_2 = []
        self.Zone_Air_Temperature_2 = []
        self.Zone_Mean_Radiant_Temperature_2 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_2 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_2 = []

        # thermal zone 3
        self.Zone_Air_Relative_Humidity_3 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_3 = []
        self.Zone_Infiltration_Mass_3 = []
        self.Zone_Mechanical_Ventilation_Mass_3 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_3 = []
        self.Zone_Air_Temperature_3 = []
        self.Zone_Mean_Radiant_Temperature_3 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_3 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_3 = []

        # thermal zone 4
        self.Zone_Air_Relative_Humidity_4 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_4 = []
        self.Zone_Infiltration_Mass_4 = []
        self.Zone_Mechanical_Ventilation_Mass_4 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_4 = []
        self.Zone_Air_Temperature_4 = []
        self.Zone_Mean_Radiant_Temperature_4 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_4 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_4 = []

        # thermal zone 5
        self.Zone_Air_Relative_Humidity_5 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_5 = []
        self.Zone_Infiltration_Mass_5 = []
        self.Zone_Mechanical_Ventilation_Mass_5 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_5 = []
        self.Zone_Air_Temperature_5 = []
        self.Zone_Mean_Radiant_Temperature_5 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_5 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_5 = []

        # thermal zone 6
        self.Zone_Air_Relative_Humidity_6 = []
        self.Zone_Windows_Total_Heat_Gain_Energy_6 = []
        self.Zone_Infiltration_Mass_6 = []
        self.Zone_Mechanical_Ventilation_Mass_6 = []
        self.Zone_Mechanical_Ventilation_Mass_Flow_Rate_6 = []
        self.Zone_Air_Temperature_6 = []
        self.Zone_Mean_Radiant_Temperature_6 = []
        self.Zone_Thermostat_Heating_Setpoint_Temperature_6 = []
        self.Zone_Thermostat_Cooling_Setpoint_Temperature_6 = []

        self.Zone_Mean_Temperature = []

        # Energy
        self.Electricity_Facility = []
        self.Electricity_HVAC = []
        self.Heating_Electricity = []
        self.Cooling_Electricity = []

        self.Electricity_Zone_1 = []
        # Time
        self.x = []

        self.T_years = []
        self.T_months = []
        self.T_days = []
        self.T_hours = []
        self.T_minutes = []
        self.T_current_times = []
        self.T_actual_date_times = []
        self.T_actual_times = []
        self.T_time_steps = []

        self.T_weekday = []
        self.T_isweekday = []
        self.T_isweekend = []
        self.T_work_time = []


        '''for DQN'''  # 未加后缀的变量都是针对于eposide
        self.state = []
        self.action = []
        self.reward = []
        self.reward_random_memory = []
        self.next_state = []
        self.done = []
        self.loss = []
        self.loss_random_memory = []
        self.Temp_mean_violation = []
        self.Temp_violation = []

    def initialize_handels(self):
        """
        This part is for ENVIRONMENT actuator handles
        """
        self.handle_Environmental_Impact_Total_CO2_Emissions_Carbon_Equivalent_Mass = -1

        self.handle_Site_Outdoor_Air_Drybulb_Temperature = -1
        self.handle_Site_Wind_Speed = -1
        self.handle_Site_Wind_Direction = -1
        self.handle_Site_Solar_Azimuth_Angle = -1
        self.handle_Site_Solar_Altitude_Angle = -1
        self.handle_Site_Solar_Hour_Angle = -1

        """
        This part is for ZONE variable handles
        """

        # thermal zone 1
        self.handle_Zone_Air_Relative_Humidity_1 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_1 = -1
        self.handle_Zone_Infiltration_Mass_1 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_1 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1 = -1
        self.handle_Zone_Air_Temperature_1 = -1
        self.handle_Zone_Mean_Radiant_Temperature_1 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_1 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_1 = -1

        # thermal zone 2
        self.handle_Zone_Air_Relative_Humidity_2 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_2 = -1
        self.handle_Zone_Infiltration_Mass_2 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_2 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_1 = -1
        self.handle_Zone_Air_Temperature_2 = -1
        self.handle_Zone_Mean_Radiant_Temperature_2 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_2 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_2 = -1

        # thermal zone 3
        self.handle_Zone_Air_Relative_Humidity_3 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_3 = -1
        self.handle_Zone_Infiltration_Mass_3 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_3 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_3 = -1
        self.handle_Zone_Air_Temperature_3 = -1
        self.handle_Zone_Mean_Radiant_Temperature_3 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_3 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_3 = -1

        # thermal zone 4
        self.handle_Zone_Air_Relative_Humidity_4 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_4 = -1
        self.handle_Zone_Infiltration_Mass_4 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_4 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_4 = -1
        self.handle_Zone_Air_Temperature_4 = -1
        self.handle_Zone_Mean_Radiant_Temperature_4 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_4 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_4 = -1

        # thermal zone 5
        self.handle_Zone_Air_Relative_Humidity_5 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_5 = -1
        self.handle_Zone_Infiltration_Mass_5 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_5 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_5 = -1
        self.handle_Zone_Air_Temperature_5 = -1
        self.handle_Zone_Mean_Radiant_Temperature_5 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_5 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_5 = -1

        # thermal zone 6
        self.handle_Zone_Air_Relative_Humidity_6 = -1
        self.handle_Zone_Windows_Total_Heat_Gain_Energy_6 = -1
        self.handle_Zone_Infiltration_Mass_6 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_6 = -1
        self.handle_Zone_Mechanical_Ventilation_Mass_Flow_Rate_6 = -1
        self.handle_Zone_Air_Temperature_6 = -1
        self.handle_Zone_Mean_Radiant_Temperature_6 = -1
        self.handle_Zone_Thermostat_Heating_Setpoint_Temperature_6 = -1
        self.handle_Zone_Thermostat_Cooling_Setpoint_Temperature_6 = -1

        # Energy handles --- meters
        self.handle_Electricity_Facility = -1
        self.handle_Electricity_HVAC = -1
        self.handle_Heating_Electricity = -1
        self.handle_Cooling_Electricity = -1

        self.handle_Electricity_Zone_1 = -1

        # Actuator handles
        self.handle_Heating_Setpoint_1 = -1
        self.handle_Cooling_Setpoint_1 = -1
        self.handle_Heating_Setpoint_2 = -1
        self.handle_Cooling_Setpoint_2 = -1
        self.handle_Heating_Setpoint_3 = -1
        self.handle_Cooling_Setpoint_3 = -1
        self.handle_Heating_Setpoint_4 = -1
        self.handle_Cooling_Setpoint_4 = -1
        self.handle_Heating_Setpoint_5 = -1
        self.handle_Cooling_Setpoint_5 = -1
        self.handle_Heating_Setpoint_6 = -1
        self.handle_Cooling_Setpoint_6 = -1

        self.handles = []
        self.is_handle = False
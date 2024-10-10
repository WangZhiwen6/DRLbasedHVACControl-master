import openstudio
import os

#将osm文件转换为idf文件
class osm2idf():

    def __init__(self, osm_file):
        self.osm_file = osm_file
        current_dir = os.getcwd()
        osm_path = os.path.join(current_dir, osm_file)
        osm_path = openstudio.path(osm_path)  # I guess this is how it wants the path for the translator
        print(osm_path)
        self.translate(osm_path)
        self.out_varbirates_specification()
#模型加载
    def translate(self, osm_path):
        # i guess this is for eliminating the conflicts between  different OP versions
        # also may cause some problems hard to resolve
        self.translator = openstudio.osversion.VersionTranslator()
        self.m = self.translator.loadModel(osm_path).get()
#输出变量设置
    def out_varbirates_specification(self):
        zones = [zone for zone in openstudio.model.getThermalZones(self.m)]
        # this is for removing all the existing variables in the original osmfile
        zones_name = []
        for i in range(len(zones)):
            zones_name.append('Thermal Zone ' + str(i+1))
        # print(zones_name)

#删除模型现有的输出变量
        [x.remove() for x in self.m.getOutputVariables()]

        """
        this is for specifying ENVIRONMENT output variables
        
        if you want to change variables, you need to change the contents of the list
        """
#设置环境输出变量
        for var in [
            "Site Outdoor Air Drybulb Temperature",
            "Site Wind Speed",
            "Site Wind Direction",
            "Site Solar Azimuth Angle",
            "Site Solar Altitude Angle",
            "Site Solar Hour Angle"
        ]:
            output_variables = openstudio.model.OutputVariable(var, self.m)
            output_variables.setKeyValue('Environment')
            output_variables.setReportingFrequency('Timestep')


        """
        this is for pecifying Thermal Zone output variables
        """

        for zone_name in zones_name:
            for var in [
                "Zone Air Relative Humidity",
                "Zone Windows Total Heat Gain Energy",
                "Zone Infiltration Mass",
                "Zone Mechanical Ventilation Mass",
                "Zone Mechanical Ventilation Mass Flow Rate",
                "Zone Air Temperature",
                "Zone Mean Radiant Temperature",
                "Zone Thermostat Heating Setpoint Temperature",
                "Zone Thermostat Cooling Setpoint Temperature"
            ]:
                output_variables = openstudio.model.OutputVariable(var, self.m)
                output_variables.setKeyValue(zone_name)
                output_variables.setReportingFrequency('Timestep')


        print(len(self.m.getOutputVariables()))#获取所有输出变量的列表的长度
        for i in self.m.getOutputVariables():
            print(i)

        self.set_timestep(12)
        self.set_run_period(1, 1, 12, 31)
        self.sava2idf()
#设置运行周期，步长，保存idf文件
    def set_run_period(self, begin_M=1, begin_D=1, end_M=12, end_D=31):#1月1到12月31

        """
        the default settings of run period is from 1.1 to 1.31
        """

        run_period = self.m.getRunPeriod()
        run_period.setBeginMonth(begin_M)
        run_period.setBeginDayOfMonth(begin_D)

        run_period.setEndMonth(end_M)
        run_period.setEndDayOfMonth(end_D)
        print(run_period)

    def set_timestep(self, step=6):#每小时步数6
        timestep = self.m.getTimestep()
        timestep.setNumberOfTimestepsPerHour(step)
        print(timestep)

    def sava2idf(self, save_path = './EPmodel/'):
        ft = openstudio.energyplus.ForwardTranslator()
        w = ft.translateModel(self.m)
        self.save_path = save_path + 'OSM_Translated_IDF'
        w.save(openstudio.path(self.save_path), True)
        print(f"the model has been successfully saved to '{self.save_path}.idf'")
        self.idf_file = self.save_path+'.idf'

# if __name__ == '__main__':
#     idf = osm2idf("C:/Users/Lee/Desktop/new_ue_room/1.5-1-test.osm")
#     idf.out_varbirates_specification()
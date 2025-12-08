#定义配置类，支持读取默认设置，#也可以通过传入参数覆盖默认设置，也可以添加新的配置项
import json 
class CONFIG:    
    def __init__(self, config_path="./config.json"):
        # 默认配置项
        try:
            with open(config_path, "r") as cf:
                self.config_default = json.load(cf)
        except FileNotFoundError:
            print(f"配置文件 {config_path} 未找到,未能读取配置文件")
            return 
        self.config=dict()
        self.get_default()
    #设置默认配置
    def get_default(self):
        #获取路径配置
        self.config.update(self.config_default["path"])
        
        #获取tiff_cut配置
        self.config.update(self.config_default["tiff_cut"])      
        
        #训练数据集配置
        train_dataset=self.config_default["train"]["train_dataset"]
        train_dataset_cfg=self.config_default["train"]["train_"+train_dataset]
        self.config.update(train_dataset_cfg)
        self.config.update({"train_dataset":train_dataset})
        
        #获取预训练模型配置
        pretrained_model_cfg=self.config_default["pretrained_model"]
        self.config.update(pretrained_model_cfg)

        #获取验证配置
        val_dataset=self.config_default["validate"]["validate_dataset"]
        val_dataset_cfg=self.config_default["validate"]["validate_"+val_dataset]
        self.config.update(val_dataset_cfg)
        
        #获取测试配置
        test_dataset=self.config_default["test"]["test_dataset"]
        test_dataset_cfg=self.config_default["test"]["test_"+test_dataset]
        self.config.update(test_dataset_cfg)
        self.config.update({"test_dataset":test_dataset})
        
        #获取可视化配置
        visualize_dataset_cfg=self.config_default["visualization"]
        self.config.update(visualize_dataset_cfg)
        
    #更新配置，可以添加新的配置项
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                self.config[key] = value

    #输出当前配置
    def display_config(self):
        for key, value in self.config.items():
            print("-----") if key.startswith("?") else None
            print(f"{key}: {value}")
            print("-----") if not key.startswith("?") else None
    
    def __getitem__(self, key):
        return self.config.get(key)
    
    def __len__(self):
        return len(self.config)
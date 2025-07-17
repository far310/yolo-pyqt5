import paho.mqtt.client as mqtt
# from getmac import get_mac_address
import os
import hashlib
import requests
import time
import hashlib
import hmac
def get_machine_id():
    with open("/etc/machine-id", "r") as f:
        return f.read().strip()
def encode_md5(file_path, salt):
    # 读取文件字节
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    # 计算文件的 MD5 哈希
    md5_hash = hashlib.md5(file_bytes).hexdigest()

    # 计算 HMAC MD5, 使用盐作为密钥
    hmac_md5 = hmac.new(salt.encode(), md5_hash.encode(), hashlib.md5).hexdigest()

    return hmac_md5
class MQTTClient:
    def __init__(self, client_id, username, password, url, port, keep_alive=60, will_topic=None, will_message=None):
        self.client_id = client_id
        self.username = username
        self.password = password
        self.url = url
        self.port = port
        self.keep_alive = keep_alive
        self.client = mqtt.Client(client_id=self.client_id)
        
        # 设置用户名和密码
        self.client.username_pw_set(username=self.username, password=self.password)

        # 设置遗嘱消息
        if will_topic and will_message:
            print(f"will_topic: {will_message}")
            self.client.will_set(will_topic+"/"+self.client_id, will_message, qos=0, retain=False)
        
        # 设置回调函数
        self.client.on_message = self.on_message

    def connect(self):
        try:
            self.client.connect(self.url, self.port, keepalive=self.keep_alive)
            self.client.loop_start()
            print(f"Connected to MQTT broker at {self.url}:{self.port} with keep alive {self.keep_alive} seconds")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            return False
        return True

    def publish(self, topic, message):
        try:
            self.client.publish(topic, message)
            print(f"Published message: '{message}' to topic: '{topic}'")
        except Exception as e:
            print(f"Failed to publish message: {e}")

    def subscribe(self, topic):
        try:
            self.client.subscribe(topic)
            print(f"Subscribed to topic: '{topic}'")
        except Exception as e:
            print(f"Failed to subscribe to topic: {e}")

    # 回调函数 - 收到消息时触发
    def on_message(self, client, userdata, message):
        topic = message.topic
        payload = message.payload.decode()

        print(f"Received message on topic: '{topic}'")
        flag =topic == ('up/'+self.client_id)
        print(f"Firmware update received: {payload}{flag }")
        # 如果是固件更新主题，执行固件更新逻辑
        if topic == ("up/"+self.client_id):
            print(f"Firmware update received: {payload}")
            
            # 假设 payload 是 JSON 字符串，包含固件下载地址和文件哈希值
            firmware_data = self.parse_firmware_payload(payload)
            if firmware_data:
                firmware_url = firmware_data.get("url")
                firmware_hash = firmware_data.get("hash")
                self.download_and_update_firmware(firmware_url, firmware_hash)

    # 解析固件更新的消息，获取固件下载地址和哈希值
    def parse_firmware_payload(self, payload):
        try:
            import json
            data = json.loads(payload)
            firmware_url = data.get("url")
            firmware_hash = data.get("hash")

            if firmware_url and firmware_hash:
                return {"url": firmware_url, "hash": firmware_hash}
            else:
                print("Invalid firmware update payload")
                return None
        except Exception as e:
            print(f"Failed to parse firmware payload: {e}")
            return None

    # 固件更新逻辑
    def download_and_update_firmware(self, firmware_url, firmware_hash):
        try:
            self.publish("receive/"+self.client_id, "开始下载固件")
            # 下载固件
            print(f"Downloading firmware from: {firmware_url}")
            response = requests.get(firmware_url, stream=True)      
            # 保存固件到本地文件
            firmware_file = "./temp/model.tflite"
            firmware_dir = os.path.dirname(firmware_file)
            print(f'firmware_dir ：{firmware_dir}')
            # 如果目录不存在，先创建目录
            if not os.path.exists(firmware_dir):
                os.makedirs(firmware_dir)  # 创建目录
                
            with open(firmware_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Firmware downloaded and saved as {firmware_file}")
            self.publish("receive/"+self.client_id, "固件下载完成")
            # 校验文件完整性（通过哈希值校验）
            if self.verify_firmware(firmware_file, firmware_hash):
                print("Firmware verification successful.")
                self.publish("receive/"+self.client_id, "校验文件完整性成功")
                # 安装固件
                self.install_firmware(firmware_file)
            else:
                self.publish("receive/"+self.client_id, "校验文件完整性失败")
                print("Firmware verification failed. Aborting update.")

        except Exception as e:
            self.publish("receive/"+self.client_id, "固件下载失败")
            print(f"Failed to download or update firmware: {e}")

    # 校验固件完整性
    def verify_firmware(self, firmware_file, expected_hash):
        try:
            # sha256 = hashlib.sha256()

            # with open(firmware_file, "rb") as f:
            #     while True:
            #         data = f.read(1024)
            #         if not data:
            #             break
            #         sha256.update(data)

            # file_hash = sha256.hexdigest()
            file_hash = encode_md5(firmware_file,'ebop-123456')
            print(f"Calculated hash: {file_hash}")
            print(f"Expected hash: {expected_hash}")

            return file_hash == expected_hash
        except Exception as e:
            print(f"Failed to verify firmware: {e}")
            return False

    # 安装固件
    def install_firmware(self, firmware_file):
        try:
            self.publish("receive/"+self.client_id, "开始安装固件")
            # 假设安装固件的逻辑是替换某个文件或执行某个安装命令
            print(f"Installing firmware: {firmware_file}")
            # TODO: 实现安装固件的逻辑
            # 备份旧文件
            oriFile = "./model/model.tflite"
             # 1. 生成备份文件名
            timestamp = int(time.time())  # 获取当前时间戳
            backup_file = f"{firmware_file}-{timestamp}.bak"  # 备份文件名
            self.publish("receive/"+self.client_id, "旧固件开始备份")
            # 2. 如果固件文件存在，重命名为备份文件
            if os.path.exists(oriFile):
                os.rename(oriFile, backup_file)
                print(f"备份现有固件为: {backup_file}")
            else:
                print(f"未找到现有固件文件: {oriFile}")
            self.publish("receive/"+self.client_id, "旧固件备份成功")
            # 3. 如果固件文件存在，重命名为备份文件
            if os.path.exists(firmware_file):
                os.rename(firmware_file, oriFile)
                print(f"将新固件修改为: {firmware_file}")
            else:
                print(f"未找到现有固件文件: {firmware_file}")
            self.publish("receive/"+self.client_id, "新固件安装成功")
            self.publish("receive/"+self.client_id, "等待重启")
             # 3. 下载新的固件文件
            #os.system("sync && reboot")  # 重启设备以应用固件（示例命令）
            print("Firmware installed successfully and device rebooted.")
        except Exception as e:
            print(f"Failed to install firmware: {e}")

    def disconnect(self):
        self.publish("willTopic/"+self.client_id, "0")
        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected from MQTT broker")


# 示例使用
if __name__ == "__main__":
    # 获取真实的 MAC 地址作为客户端 ID
    mqtt_client_id = get_machine_id()

    # MQTT 配置
    mqtt_username = "ebop"
    mqtt_password = "ebop-123456"
    mqtt_url = "192.168.0.186"
    mqtt_port = 1883
    mqtt_keep_alive = 20  # 设置心跳间隔为60秒

    # 遗嘱消息配置
    will_topic = "willTopic"
    will_message = "0"

    # 创建 MQTT 客户端实例并配置心跳与遗嘱
    mqtt_client = MQTTClient(mqtt_client_id, mqtt_username, mqtt_password, mqtt_url, mqtt_port,
                             keep_alive=mqtt_keep_alive, will_topic=will_topic, will_message=will_message)

    # 连接到 MQTT 服务器
    if mqtt_client.connect():
        # 订阅主题1
        mqtt_client.subscribe("up/"+mqtt_client_id)

        # 发布消息
        mqtt_client.publish("willTopic/"+mqtt_client_id, "1")
        # 让程序运行一段时间以接收消息
        try:
            while True:
                pass
        except KeyboardInterrupt:
            pass

        # 断开连接
        mqtt_client.disconnect()
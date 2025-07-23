import struct
from periphery import Serial, SerialError
import threading
import argparse

# 定义类名与字节值的对应关系
CLASS_MAPPING = {
    "vanilla_bliss": 0x01,
    "strawberry_banana": 0x02,
    "snickers": 0x03,
    "smores": 0x04,
    "pbcup": 0x05,
    "oreo": 0x06,
    "mango": 0x07,
    "cotton_candy": 0x08,
    "cool_mint_chip": 0x09,
    "cinnamon_churro": 0x0A,
    "chocolate_chill": 0x0B,
    "acal_berry_blast": 0x0C
}

class UARTCommunication:
    def __init__(self, device, baudrate=115200, databits=8, parity="none", stopbits=1, xonxoff=False, rtscts=False):
         # 申请串口资源并设置串口参数
        # •	device: 串口设备路径，例如 /dev/ttyS3
        # •	baudrate: 波特率，例如 115200
        # •	databits: 数据位，例如 8
        # •	parity: 校验位，例如 "none"
        # •	stopbits: 停止位，例如 1
        # •	xonxoff: 是否使用软件流控制，例如 False
        # •	rtscts: 是否使用硬件流控制，例如 False
        # •	data_to_send: 要发送的数据，以字节流的形式，例如 b"python-periphery!\n"
        # •	timeout: 读取超时时间，例如 1 秒
        self.device = device
        self.baudrate = baudrate
        self.databits = databits
        self.parity = parity
        self.stopbits = stopbits
        self.xonxoff = xonxoff
        self.rtscts = rtscts
        self.serial = None
        self.running = False
        self.callback = None

    def open(self):
        self.serial = Serial(
            self.device,
            baudrate=self.baudrate,
            databits=self.databits,
            parity=self.parity,
            stopbits=self.stopbits,
            xonxoff=self.xonxoff,
            rtscts=self.rtscts,
        )

    def close(self):
        self.running = False
        if self.serial:
            self.serial.close()

    def send_data(self, class_name,header=0x05,type_byte=0x5C,length=0x0001):
        if class_name not in CLASS_MAPPING:
            print(f"类名 {class_name} 不存在于 CLASS_MAPPING 中")
            return
        command = CLASS_MAPPING[class_name]
        data_to_send = struct.pack('!BBH B', header, type_byte, length, command)
        
        print(f"发送的数据: {data_to_send}")
        self.serial.write(data_to_send)
    def read_data(self):
        # 注：Python读取出来的数据类型为：bytes
        # 解析接收的数据包
        buf = self.serial.read(128, 1)
        if len(buf) >= 5:
            header, type_byte, length, command = struct.unpack('!BBH B', buf)
            # 打印解析后的数据
            print(f"解析后的数据: Header={header}, Type={type_byte}, Length={length}, Command={command}")
            # 打印原始数据
            print("接收的原始数据:\n", buf)
            return command
        else:
            return ''
    def receive_data(self, expected_length=5):
        try:
            while self.running:
                buf = self.serial.read(expected_length, 1)
                if len(buf) == expected_length:
                    self.unpack_data(buf)
                else:
                    print(f"读取的数据长度不足 {expected_length} 字节:", len(buf))
        except SerialError as e:
            print("串口通信错误:", e)

    def unpack_data(self, buf):
        try:
            header, type_byte, length, command = struct.unpack('!BBH B', buf)
            print(f"解析后的数据: Header={header}, Type={type_byte}, Length={length}, Command={command}")
            if self.callback:
                self.callback(header, type_byte, length, command)
        except struct.error as e:
            print(f"解包错误: {e}")

    def start_receiving(self, callback):
        self.running = True
        self.callback = callback
        thread = threading.Thread(target=self.receive_data)
        thread.start()

def handle_received_data(header, type_byte, length, command):
    print(f"回调函数处理接收到的数据: Header={header}, Type={type_byte}, Length={length}, Command={command}")

def main():
    parser = argparse.ArgumentParser(description="UART communication test")
    parser.add_argument('--device', type=str, required=True, help='Serial device path, e.g., /dev/ttyS3')
    parser.add_argument('--class_name', type=str, required=True, choices=CLASS_MAPPING.keys(), help='Class name to send')
    args = parser.parse_args()

    uart = UARTCommunication(args.device)
    uart.open()
    uart.start_receiving(handle_received_data)
    uart.send_data(args.class_name)

    try:
        while True:
            pass  # 保持主线程运行
    except KeyboardInterrupt:
        uart.close()

if __name__ == "__main__":
    main()
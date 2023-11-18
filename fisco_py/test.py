import sys, os, torch
#一定要加下面这两个东西


# sys.path.append("/data/wangweicheng/czj/Vateer-FL/fisco_py")
# 将当前工作目录切换到fisco_py所在的目录
# os.chdir('./fisco_py')
from client.bcosclient import BcosClient
from client.datatype_parser import DatatypeParser
# from blockchain.IPFS import IPFS
# from fisco_py.client.signer_impl import Signer_ECDSA #这个不行
from client.signer_impl import Signer_ECDSA   #这个才可以
# from models.CNN import Mnist_CNN
client = BcosClient()
print(client.getBlockNumber())

# net = Mnist_CNN()
# torch.save(net.state_dict(), "../cache/save.pkl")
# model = torch.load("../cache/save.pkl")
# client.finish()
# exit()
# abi_file = "./contracts/Aggregrate.abi"
abi_file = "./contracts/HelloWorld.abi"
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi

# to_address = "0xc6ed6631d2fadbeb466c8fb87fb125eae0333183"
to_address = "0xa51ed90659fe72ef5190b599dea491f19acd635b"
args=[]
# receipt = client.sendRawTransactionGetReceipt(to_address, contract_abi, "get_threshold", args, from_account_signer = Signer_ECDSA.from_key_file("bin/accounts/czj.keystore", "123456"))
# Signer_ECDSA.from_key_file("bin/accounts/czj.keystore", "123456")
# receipt = client.sendRawTransactionGetReceipt(to_address, contract_abi, "init")
# receipt = client.sendRawTransactionGetReceipt(to_address, contract_abi, "set_threshold", args = [100])
# print(f"receipt = client.sendRawTransactionGetReceipt({to_address}, {contract_abi}, 'set',['sdf'],gasPrice=30000000,from_account_signer=None)")
print(contract_abi)
receipt = client.sendRawTransactionGetReceipt(to_address, contract_abi, "set",['sdf'],gasPrice=30000000,from_account_signer=None)
print(receipt)
# receipt = client.call(to_address, contract_abi, "get_threshold", args)
# print(int(receipt["output"],16)) #要结果
# ipfs = IPFS()
# # res = ipfs.push_local_file("../cache/model.pkl")
# # print(res)
# ipfs.download_loacl_file("QmPwKSWLL237i1DbYvr1Y28XCNn4X5CX7hGa6LkVfg43Pn","/data/wangweicheng/czj/Vateer-FL/cache/down.pkl")
client.finish()

# receipt = temp.sendRawTransactionGetReceipt("0xa51ed90659fe72ef5190b599dea491f19acd635b", [{'constant': False, 'inputs': [{'name': 'n', 'type': 'string'}], 'name': 'set', 'outputs': [], 'payable': False, 'stateMutability': 'nonpayable', 'type': 'function'}, {'constant': True, 'inputs': [], 'name': 'get', 'outputs': [{'name': '', 'type': 'string'}], 'payable': False, 'stateMutability': 'view', 'type': 'function'}, {'inputs': [], 'payable': False, 'stateMutability': 'nonpayable', 'type': 'constructor'}, {'anonymous': False, 'inputs': [{'indexed': False, 'name': 'newname', 'type': 'string'}], 'name': 'onset', 'type': 'event', 'topic': '0xafb180742c1292ea5d67c4f6d51283ecb11e49f8389f4539bef82135d689e118'}], set, ['sdf'], None, 30000000, from_account_signer=None)
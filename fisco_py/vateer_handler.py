from client.bcosclient import BcosClient
import fire
from client.signer_impl import Signer_ECDSA
from client.datatype_parser import DatatypeParser


def main(address, func, account_name=None, args = None):
    if account_name:
        account_name = Signer_ECDSA.from_key_file("bin/accounts/{}.keystore".format(account_name), "123456")
    if args:
        args = list(args)
    else:
        args = []
    address = address[1:]
    client = BcosClient()
    abi_file = "./contracts/Aggregrate.abi"
    data_parser = DatatypeParser()
    data_parser.load_abi_file(abi_file)
    contract_abi = data_parser.contract_abi
    res = ""
    if func == "test":
        abi_file = "./contracts/HelloWorld.abi"
        data_parser = DatatypeParser()
        data_parser.load_abi_file(abi_file)
        contract_abi = data_parser.contract_abi
        res = client.sendRawTransactionGetReceipt(hex(int(address, 16)), contract_abi, "set",['sdfe'] ,from_account_signer=account_name)
    elif func == "set_threshold":
        client.sendRawTransactionGetReceipt(hex(int(address, 16)), contract_abi, "init")
        res = client.sendRawTransactionGetReceipt(hex(int(address, 16)), contract_abi, "set_threshold", args ,from_account_signer=account_name)
    elif func == "send_gradient":
        res = client.sendRawTransactionGetReceipt(hex(int(address, 16)), contract_abi, "send_gradient", args ,from_account_signer=account_name)
    elif func == "get_gradient":
        res = client.sendRawTransactionGetReceipt(hex(int(address, 16)), contract_abi, "get_gradient", args ,from_account_signer=account_name)
        txhash = res['transactionHash']
        txresponse = client.getTransactionByHash(txhash)
        inputresult = data_parser.parse_transaction_input(txresponse['input'])
        outputresult = data_parser.parse_receipt_output(inputresult['name'], res['output'])
        res = outputresult[0]

    print(res)


if __name__ == "__main__":
    fire.Fire(main)


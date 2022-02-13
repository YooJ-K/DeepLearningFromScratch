from mulLayer import *
from addLayer import *

def mul():
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(price)

    # 역전파
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num, dtax)

    # 220.00000000000003
    # 2.2 110.00000000000001 200

def mul_and_add():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    # 역전파
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(price)
    print(dapple_num, dapple, dorange, dorange_num, dtax)

    # 715.0000000000001
    # 110.00000000000001 2.2 3.3000000000000003 165.0 650

mul_and_add()
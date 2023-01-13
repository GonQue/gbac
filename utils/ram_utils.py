def get_num_layers(glimpse_size):
    if 0 < glimpse_size <= 25:
        return 1 
    elif 25 < glimpse_size <= 75:
        return 2
    else:
        return 3


def get_kernels_and_strides(glimpse_size):
    if 0 < glimpse_size <= 15: # 10 
        return 4, 2
    
    elif 15 < glimpse_size <= 25: # 20
        return 6, 4

    elif 25 < glimpse_size <= 35: # 30
        return 5, 2, 4, 2

    elif 35 < glimpse_size <= 55: # 40, 50
        return 6, 3, 4, 2

    elif 55 < glimpse_size <= 65: # 60
        return 8, 4, 4, 2

    elif 65 < glimpse_size <= 75: # 70
        return 8, 4, 6, 2

    elif 75 < glimpse_size <= 85: # 80
        return 8, 4, 4, 2, 4, 1

    elif 85 < glimpse_size <= 96: # 90
        return 8, 4, 4, 2, 3, 1

    elif 96 < glimpse_size <= 105: # 100
        return 10, 5, 4, 2, 2, 1

    elif 105 < glimpse_size <= 125: # 110, 120
        return 10, 5, 4, 2, 3, 1

    elif 125 < glimpse_size <= 135: # 130
        return 10, 5, 6, 2, 3, 1

    elif 135 < glimpse_size <= 145: # 140
        return 8, 4, 6, 3, 3, 1

    elif glimpse_size > 145: # 150, 160
        return 8, 4, 6, 3, 4, 1


def get_num_layers_vae(glimpse_size):
    if 0 < glimpse_size <= 35:
        return 2
    elif 35 < glimpse_size <= 75:
        return 3
    else:
        return 4


def get_kernels_and_strides_vae(glimpse_size):
    if 0 < glimpse_size <= 15: # 10  7x7, 5x5
        return 4, 1, 3, 1
    
    elif 15 < glimpse_size <= 25: # 20  8x8, 5x5
        return 6, 2, 4, 1

    elif 25 < glimpse_size <= 35: # 30  12x12, 6x6
        return 7, 2, 2, 2

    elif 35 < glimpse_size <= 45: # 40  12x12, 9x9, 6x6
        return 6, 3, 4, 1, 4, 1

    elif 45 < glimpse_size <= 55: # 50  15x15, 7x7, 6x6
        return 6, 3, 3, 2, 2, 1

    elif 55 < glimpse_size <= 75: # 60  14x14, 7x7, 6x6
        return 8, 4, 2, 2, 2, 1   # 70  16x16, 8x8, 7x7

    elif 75 < glimpse_size <= 85: # 80  39x39, 18x18, 8x8, 7x7
        return 4, 2, 4, 2, 4, 2, 2, 1

    elif 85 < glimpse_size <= 95: # 90  44x44, 21x21, 9x9, 7x7
        return 4, 2, 4, 2, 4, 2, 3, 1

    elif 95 < glimpse_size <= 105: # 100  48x48, 23x23, 10x10, 7x7
        return 6, 2, 4, 2, 4, 2, 4, 1

    elif 105 < glimpse_size <= 115: # 110  53x53, 25x25, 11x11, 8x8
        return 5, 2, 4, 2, 4, 2, 4, 1  

    elif 115 < glimpse_size <= 125: # 120  58x58, 28x28, 13x13, 8x8
        return 6, 2, 4, 2, 4, 2, 6, 1  

    elif 125 < glimpse_size <= 135: # 130  63x63, 30x30, 13x13, 8x8
        return 6, 2, 5, 2, 5, 2, 6, 1

    elif 135 < glimpse_size <= 155: # 140  68x68, 32x32, 14x14, 8x8
        return 6, 2, 6, 2, 6, 2, 7, 1 # 150  73x73, 34x34, 15x15, 9x9

    elif glimpse_size > 155: # 160  77x77, 35x35, 15x15, 9x9
        return 8, 2, 8, 2, 6, 2, 7, 1


# 96


# 48

# 24
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K


class HourglassNet:
    # Adopted from https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/blob/master/src/net/hg_blocks.py
    def __init__(self, num_classes, num_stacks, num_channels, inres, outres):
        self.num_classes = num_classes
        self.num_channels = num_channels

        input = Input(shape=(inres[0], inres[1], 3))

        front_features = self.create_front_module(input)

        head_next_stage = front_features

        outputs = []
        for i in range(num_stacks):
            head_next_stage, head_to_loss = self.hourglass_module(
                head_next_stage, i)
            outputs.append(head_to_loss)

        self.model = Model(inputs=input, outputs=outputs)

    def get_model(self):
        return self.model

    def hourglass_module(self, bottom, hgid):
        # create left features , f1, f2, f4, and f8
        left_features = self.create_left_half_blocks(bottom, hgid)

        # create right features, connect with left features
        rf1 = self.create_right_half_blocks(left_features, hgid)

        # add 1x1 conv with two heads, head_next_stage is sent to next stage
        # head_parts is used for intermediate supervision
        head_next_stage, head_parts = self.create_heads(bottom, rf1, hgid)

        return head_next_stage, head_parts

    def bottleneck_block(self, bottom, num_out_channels, block_name):
        # skip layer
        if K.int_shape(bottom)[-1] == num_out_channels:
            _skip = bottom
        else:
            _skip = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                           name=block_name + 'skip')(bottom)

        # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
        _x = Conv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                    name=block_name + '_conv_1x1_x1')(bottom)
        _x = BatchNormalization()(_x)
        _x = Conv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                    name=block_name + '_conv_3x3_x2')(_x)
        _x = BatchNormalization()(_x)
        _x = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                    name=block_name + '_conv_1x1_x3')(_x)
        _x = BatchNormalization()(_x)
        _x = Add(name=block_name + '_residual')([_skip, _x])

        return _x

    def create_front_module(self, input):
        # front module, input to 1/4 resolution
        # 1 7x7 conv + maxpooling
        # 3 residual block

        _x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1_x1')(
            input)
        _x = BatchNormalization()(_x)

        _x = self.bottleneck_block(
            _x, self.num_channels // 2, 'front_residual_x1')
        _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

        _x = self.bottleneck_block(
            _x, self.num_channels // 2, 'front_residual_x2')
        _x = self.bottleneck_block(_x, self.num_channels, 'front_residual_x3')

        return _x

    def create_left_half_blocks(self, bottom, hglayer):
        # create left half blocks for hourglass module
        # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

        hgname = 'hg' + str(hglayer)

        f1 = self.bottleneck_block(bottom, self.num_channels, hgname + '_l1')
        _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

        f2 = self.bottleneck_block(_x, self.num_channels, hgname + '_l2')
        _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

        f4 = self.bottleneck_block(_x, self.num_channels, hgname + '_l4')
        _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

        f8 = self.bottleneck_block(_x, self.num_channels, hgname + '_l8')

        return (f1, f2, f4, f8)

    def connect_left_to_right(self, left, right, name):
        '''
        :param left: connect left feature to right feature
        :param name: layer name
        :return:
        '''
        # left -> 1 bottlenect
        # right -> upsampling
        # Add   -> left + right

        _xleft = self.bottleneck_block(
            left, self.num_channels, name + '_connect')
        _xright = UpSampling2D()(right)
        add = Add()([_xleft, _xright])
        out = self.bottleneck_block(
            add, self.num_channels, name + '_connect_conv')
        return out

    def bottom_layer(self, lf8, hgid):
        # blocks in lowest resolution
        # 3 bottlenect blocks + Add

        lf8_connect = self.bottleneck_block(
            lf8, self.num_channels, str(hgid) + "_lf8")

        _x = self.bottleneck_block(
            lf8, self.num_channels, str(hgid) + "_lf8_x1")
        _x = self.bottleneck_block(
            _x, self.num_channels, str(hgid) + "_lf8_x2")
        _x = self.bottleneck_block(
            _x, self.num_channels, str(hgid) + "_lf8_x3")

        rf8 = Add()([_x, lf8_connect])

        return rf8

    def create_right_half_blocks(self, leftfeatures, hglayer):
        lf1, lf2, lf4, lf8 = leftfeatures

        rf8 = self.bottom_layer(lf8, hglayer)

        rf4 = self.connect_left_to_right(
            lf4, rf8, 'hg' + str(hglayer) + '_rf4')

        rf2 = self.connect_left_to_right(
            lf2, rf4, 'hg' + str(hglayer) + '_rf2')

        rf1 = self.connect_left_to_right(
            lf1, rf2, 'hg' + str(hglayer) + '_rf1')

        return rf1

    def create_heads(self, prelayerfeatures, rf1, hgid):
        # two head, one head to next stage, one head to intermediate features
        head = Conv2D(self.num_channels, kernel_size=(1, 1), activation='relu', padding='same', name=str(hgid) + '_conv_1x1_x1')(
            rf1)
        head = BatchNormalization()(head)

        # for head as intermediate supervision, use 'linear' as activation.
        head_parts = Conv2D(self.num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                            name=str(hgid) + '_conv_1x1_parts')(head)

        # use linear activation
        head = Conv2D(self.num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                      name=str(hgid) + '_conv_1x1_x2')(head)
        head_m = Conv2D(self.num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                        name=str(hgid) + '_conv_1x1_x3')(head_parts)

        head_next_stage = Add()([head, head_m, prelayerfeatures])
        return head_next_stage, head_parts

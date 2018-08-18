#include "../device/hw_param.cl"

#define NUM_CONFIG_ITEM  26

/*
// VGG-16 Configuration
unsigned layer_config[][NUM_CONFIG_ITEM] = {{0,
							224, 224, 3, 3, 3, 3, 64, 64,
							0,
							224, 224, 64, 1, 1, 0, 1,
							0, 224, 224, 64, 0, 0,
							0,
							1},//Layer-1 (conv1_1)
							{0,
							224, 224, 64, 3, 3, 64, 64, 64,
							1,
							224, 224, 64, 1, 1, 0, 1,
							1, 112, 112, 64, 2, 2,
							0,
							0},//Layer-2 (conv1_2)
							{0,
							112, 112, 64, 3, 3, 64, 128, 128,
							0,
							112, 112, 128, 1, 1, 0, 1,
							0, 112, 112, 128, 0, 0,
							0,
							1},//Layer-3 (conv2_1)
							{0,
							112, 112, 128, 3, 3, 128, 128, 128,
							1,
							112, 112, 128, 1, 1, 0, 1,
							1, 56, 56, 128, 2, 2,
							0,
							0},//Layer-4 (conv2_2)
							{0,
							56, 56, 128, 3, 3, 128, 256, 256,
							0,
							56, 56, 256, 1, 1, 0, 1,
							0, 56, 56, 256, 0, 0,
							0,
							1},//Layer-5 (conv3_1)
							{0,
							56, 56, 256, 3, 3, 256, 256, 256,
							1,
							56, 56, 256, 1, 1, 0, 1,
							0, 56, 56, 256, 0, 0,
							0,
							0},//Layer-6 (conv3_2)
							{0,
							56, 56, 256, 3, 3, 256, 256, 256,
							0,
							56, 56, 256, 1, 1, 0, 1,
							1, 28, 28, 256, 2, 2,
							0,
							1},//Layer-7 (conv3_3)
							{0,
							28, 28, 256, 3, 3, 256, 512, 512,
							1,
							28, 28, 512, 1, 1, 0, 1,
							0, 28, 28, 512, 0, 0,
							0,
							0},//Layer-8  (conv4_1)
							{0,
							28, 28, 512, 3, 3, 512, 512, 512,
							0,
							28, 28, 512, 1, 1, 0, 1,
							0, 28, 28, 512, 0, 0,
							0,
							1},//Layer-9  (conv4_2)
							{0,
							28, 28, 512, 3, 3, 512, 512, 512,
							1,
							28, 28, 512, 1, 1, 0, 1,
							1, 14, 14, 512, 2, 2,
							0,
							0},//Layer-10 (conv4_3)
							{0,
							14, 14, 512, 3, 3, 512, 512, 512,
							0,
							14, 14, 512, 1, 1, 0, 1,
							0, 14, 14, 512, 0, 0,
							0,
							1},//Layer-11  (conv5_1)
							{0,
							14, 14, 512, 3, 3, 512, 512, 512,
							1,
							14, 14, 512, 1, 1, 0, 1,
							0, 14, 14, 512, 0, 0,
							0,
							0},//Layer-12  (conv5_2)
							{0,
							14, 14, 512, 3, 3, 512, 512, 512,
							0,
							14, 14, 512, 1, 1, 0, 1,
							1, 7, 7, 512, 2, 2,
							0,
							2},//Layer-13  (conv5_3)    Note: for last conv layer, outputs are write to fc buffer
							{1,
							28, 28, 512, 7, 7, 512, 4096, 4096,
							2,
							4, 4, 4096, 7, 0, 0, 1,
							0, 4, 4, 4096, 0, 0,
							0,
							3},//Layer-14  (fc6)							
							{1,
							4, 4, 4096, 1, 1, 4096, 4096, 4096,
							3,
							4, 4, 4096, 1, 0, 0, 1,
							0, 4, 4, 4096, 0, 0,
							0,
							2},//Layer-15  (fc7)
							{1,
							4, 4, 4096, 1, 1, 4096, 1024, 1024,
							2,
							4, 4, 1024, 1, 0, 0, 0,
							0, 4, 4, 1024, 0, 0,
							0,
							3}//Layer-16  (fc8)		
							};

char precision_config[][3] ={{7,  0, -2},//Layer-1
							{ 8, -2, -5},//Layer-2
							{ 8, -5, -5},//Layer-3
							{ 8, -5, -6},//Layer-4
							{ 7, -6, -7},//Layer-5
							{ 8, -7, -7},//Layer-6
							{ 8, -7, -7},//Layer-7
							{ 8, -7, -6},//Layer-8
							{ 8, -6, -5},//Layer-9
							{ 8, -5, -5},//Layer-10
							{ 9, -5, -4},//Layer-11
							{ 9, -4, -3},//Layer-12
							{ 8, -3, -2},//Layer-13
							{ 8, -2,  0},//Layer-14
							{ 7,  0,  2},//Layer-15
							{ 7,  2,  2}//Layer-16
							};

unsigned input_config[4] = {224, 224, 3, 16};

//unsigned output_config[3] = {112, 112, 64};//Layer-2

unsigned output_config[3] = {1, 1, 1024};//Layer-16
*/

/*
// Alexnet Configuration batch=16
unsigned layer_config[][NUM_CONFIG_ITEM] = {{0,
							227, 227, 3, 11, 11, 3, 96, 96,
							0,
							55, 55, 96, 4, 0, 0, 1,
							1, 27, 27, 96, 3, 2,
							1,
							1},//Layer-1
							{0,
							27, 27, 96, 5, 5, 48, 256, 256,
							0,
							27, 27, 256, 1, 2, 1, 1,
							1, 13, 13, 256, 3, 2,
							1,
							1},//Layer-2
							{0,
							13, 13, 256, 3, 3, 256, 384, 384,
							0,
							13, 13, 384, 1, 1, 0, 1,
							0, 13, 13, 384, 0, 0,
							0,
							1},//Layer-3
							{0,
							13, 13, 384, 3, 3, 192, 384, 384,
							1,
							13, 13, 384, 1, 1, 1, 1,
							0, 13, 13, 384, 0, 0,
							0,
							0},//Layer-4
							{0,
							13, 13, 384, 3, 3, 192, 256, 256,
							0,
							13, 13, 256, 1, 1, 1, 1,
							1, 6, 6, 256, 3, 2,
							0,
							2},//Layer-5  Note: for last conv layer, outputs are write to fc buffer
							{1,
							24, 24, 256, 6, 6, 256, 4096, 4096,  // Note: The input size (dim1/dim2) is the combined data size (batched)
							2,
							4, 4, 4096, 6, 0, 0, 1,
							0, 4, 4, 4096, 0, 0,
							0,
							3},//Layer-6 fc
							{1,
							4, 4, 4096, 1, 1, 4096, 4096, 4096,
							3,
							4, 4, 4096, 1, 0, 0, 1,
							0, 4, 4, 4096, 0, 0,
							0,
							2},//Layer-7 fc
							{1,
							4, 4, 4096, 1, 1, 4096, 1024, 1024,
							2,
							4, 4, 1024, 1, 0, 0, 0,
							0, 4, 4, 1024, 0, 0,
							0,
							3}//Layer-8 fc
							};

char precision_config[][3] ={{8,  0, -4},//Layer-1
							{ 8,  0, -2},//Layer-2
							{ 8,  0, -1},//Layer-3
							{ 8, -1, -1},//Layer-4
							{ 8, -1, -1},//Layer-5
							{11, -1,  0},//Layer-6
							{10,  0,  2},//Layer-7
							{10,  2,  2}//Layer-8
							};

unsigned input_config[5] = {227, 227, 3, 16}; //original image size(dim1, dim2, dim3), batch size

//unsigned output_config[3] = {27, 27, 96};//Layer-1

//unsigned output_config[3] = {6, 6, 256};//Layer-5

//unsigned output_config[3] = {1, 1, 4096};//Layer-6

unsigned output_config[3] = {1, 1, 1024};//Layer-8  Note: only one result is extracted and verified
*/


/* ------------------------------------------------------------------------
 * 
 * The following configurations are used for development and test only
 *
 * ------------------------------------------------------------------------
 */

 /*	
// Test FC only (AlexNet fc6-fc8)
unsigned layer_config[][NUM_CONFIG_ITEM] = {{1,
							6, 6, 256, 6, 6, 256, 4096, 4096,  // Note: The input size (dim1/dim2) is the combined data size (batched)
							2,
							1, 1, 4096, 6, 0, 0, 1,
							0, 1, 1, 4096, 0, 0,
							0,
							3},//Layer-6 fc
							{1,
							1, 1, 4096, 1, 1, 4096, 4096, 4096,
							3,
							1, 1, 4096, 1, 0, 0, 1,
							0, 1, 1, 4096, 0, 0,
							0,
							2},//Layer-7 fc
							{1,
							1, 1, 4096, 1, 1, 4096, 1024, 1024,
							2,
							1, 1, 1024, 1, 0, 0, 0,
							0, 1, 1, 1024, 0, 0,
							0,
							3}//Layer-8 fc
							};
							
char precision_config[][3] ={{11, -1,  0},//Layer-6
							{10,  0,  2},//Layer-7
							{10,  2,  2}//Layer-8
							};
							
unsigned input_config[3] = {6, 6, 256}; //original image size

unsigned output_config[3] = {1, 1, 1024};//Layer-8
*/


/*
// Test Conv(Relu) without PooL and LRN and batch=1
// Alexnet Configuration
unsigned layer_config[][NUM_CONFIG_ITEM] = {{0,
							227, 227, 3, 11, 11, 3, 96, 96,
							0,
							55, 55, 96, 4, 0, 0, 0, // turn relu on if needed
							0, 55, 55, 96, 3, 2, // Note: memWR share the same params with pool, these params need also to be changed
							0,
							1}//Layer-1
							};

unsigned input_config[5] = {227, 227, 3, 1}; //original image size(dim1, dim2, dim3), batch size

unsigned output_config[3] = {55, 55, 96};//Layer-1

char precision_config[][3] ={{8, 0, -4}//Layer-1
							};
*/


/*
// Test Layer-1 batch=1
// Alexnet Configuration
unsigned layer_config[][NUM_CONFIG_ITEM] = {{0,
							227, 227, 3, 11, 11, 3, 96, 96,
							0,
							55, 55, 96, 4, 0, 0, 1,
							1, 27, 27, 96, 3, 2, // Note: memWR share the same params with pool, these params need also to be changed
							1,
							1}//Layer-1
							};

unsigned input_config[5] = {227, 227, 3, 1}; //original image size(dim1, dim2, dim3), batch size

unsigned output_config[3] = {27, 27, 96};//Layer-1

char precision_config[][3] ={{8, 0, -4}//Layer-1
							};
*/

/*
// Test Layer-1 batch=1
// VGG-16 Configuration
unsigned layer_config[][NUM_CONFIG_ITEM] = {{0,
							224, 224, 3, 3, 3, 3, 64, 64,
							0,
							224, 224, 64, 1, 1, 0, 1,
							0, 224, 224, 64, 0, 0,
							0,
							1}//Layer-1 (conv1_1)
							};

unsigned input_config[4] = {224, 224, 3, 1};

unsigned output_config[3] = {224, 224, 64};//Layer-1

char precision_config[][3] ={{7,  0, -2}//Layer-1
							};
*/

#if 0
// Test with batch=1
// Alexnet Configuration
unsigned layer_config[][NUM_CONFIG_ITEM] =
    {{0,
      227, 227, 3, 11, 11, 3, 96, 96,
      0,
      55, 55, 96, 4, 0, 0, 1,
      1, 27, 27, 96, 3, 2,
      1,
      1},//Layer-1
     {0,
      27, 27, 96, 5, 5, 48, 256, 256,
      0,
      27, 27, 256, 1, 2, 1, 1,
      1, 13, 13, 256, 3, 2,
      1,
      1},//Layer-2
     {0,
      13, 13, 256, 3, 3, 256, 384, 384,
      0,
      13, 13, 384, 1, 1, 0, 1,
      0, 13, 13, 384, 0, 0,
      0,
      1},//Layer-3
     {0,
      13, 13, 384, 3, 3, 192, 384, 384,
      1,
      13, 13, 384, 1, 1, 1, 1,
      0, 13, 13, 384, 0, 0,
      0,
      0},//Layer-4
     {0,
      13, 13, 384, 3, 3, 192, 256, 256,
      0,
      13, 13, 256, 1, 1, 1, 1,
      1, 6, 6, 256, 3, 2,
      0,
      2},//Layer-5  Note: for last conv layer, outputs are write to fc buffer
     {1,
      6, 6, 256, 6, 6, 256, 4096, 4096,  // Note: The input size (dim1/dim2) is the combined data size (batched)
      2,
      1, 1, 4096, 6, 0, 0, 1,
      0, 1, 1, 4096, 0, 0,
      0,
      3},//Layer-6 fc
     {1,
      1, 1, 4096, 1, 1, 4096, 4096, 4096,
      3,
      1, 1, 4096, 1, 0, 0, 1,
      0, 1, 1, 4096, 0, 0,
      0,
      2},//Layer-7 fc
     {1,
      1, 1, 4096, 1, 1, 4096, 1024, 1024,
      2,
      1, 1, 1024, 1, 0, 0, 0,
      0, 1, 1, 1024, 0, 0,
      0,
      3}//Layer-8 fc
    };

char precision_config[][3] ={{8,  0, -4},//Layer-1
                             { 8,  0, -2},//Layer-2
                             { 8,  0, -1},//Layer-3
                             { 8, -1, -1},//Layer-4
                             { 8, -1, -1},//Layer-5
                             {11, -1,  0},//Layer-6
                             {10,  0,  2},//Layer-7
                             {10,  2,  2}//Layer-8
};

//unsigned input_config[5] = {227, 227, 3, 1}; //original image size(dim1, dim2, dim3), batch size

//unsigned output_config[3] = {27, 27, 96};//Layer-1
//unsigned output_config[3] = {55, 55, 96};//Layer-1

//unsigned output_config[3] = {13, 13, 256};//Layer-2

//unsigned output_config[3] = {6, 6, 256};//Layer-5

//unsigned output_config[3] = {1, 1, 4096};//Layer-6

unsigned output_config[3] = {1, 1, 1024};//Layer-8  Note: only one result is extracted and verified

#endif

#if 0

enum config_item {
  layer_type, // "0" -> conv, "1" -> fc
  // memRd params
  data_w, data_h, data_n, weight_w, weight_h, weight_n, weight_m, bias_size,
  memrd_src, // 0 -> data_buf, 1 -> output_buf, 2 -> output2_buf, 3 -> fc_1_buf, 4 -> fc_2_buf
  // Convolution params
  conv_x, conv_y, conv_z, conv_stride, conv_padding, conv_split, conv_relu,
  // Pooling params
  pool_on, /* 0 -> off, 1 -> max, 2 -> avg */ pool_x, pool_y, pool_z, pool_size, pool_stride,
  normalization, // 0 -> off, 1 -> lrn, 2 -> batchnorm
  shortcut_src, // 4 -> off, 0 -> data_buf, 1 -> output_buf, 2 -> output2_buf
  memwr_dst // 0 -> data_buf, 1 -> output_buf, 2 -> output2_buf, 3 -> fc_1_buf, 4 -> fc_2_buf
};
#endif

unsigned layer_config[][NUM_CONFIG_ITEM] =
    {{0, /* layer_type */
      32, 32, 3, 64, 3, 3, 3, 64, /* memRd params */
      0, /* memrd_src */
      32, 32, 64, 1, 1, 0, 1, /* conv params */
      0, 27, 27, 96, 3, 2, /* pooling */
      2, /* normalization */
      4, /* shortcut */
      1}, /* memwr_dst */
     
      /* layer 1 */
	  
	  /* Basic Block 0 */
     {0, /* layer_type */
      32, 32, 64, 64, 64, 3, 3, 64, /* memRd params */
      1, /* memrd_src */
      32, 32, 64, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      0}, /* memwr_dst */
     {0, /* layer_type */
      32, 32, 64, 64, 64, 3, 3, 64, /* memRd params */
      0, /* memrd_src */
      32, 32, 64, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2, /* normalization */
      1, /* shortcut */
      2}, /* memwr_dst */
	  
	  /* Basic Block 1 */
     {0, /* layer_type */
      32, 32, 64, 64, 64, 3, 3, 64, /* memRd params */
      2, /* memrd_src */
      32, 32, 64, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      1}, /* memwr_dst */
     {1, /* layer_type */
      32, 32, 64, 64, 64, 3, 3, 64, /* memRd params */
      1, /* memrd_src */
      32, 32, 64, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2, /* normalization */
      2, /* shortcut */
      0}, /* memwr_dst */
      
      /* layer 2 */
	  /* Basic Block 0 */

	 /* shortcut */
	 {0, /* layer_type */
      32, 32, 64, 128, 64, 1, 1, 128, /* memRd params */
      0, /* memrd_src */
      16, 16, 128, 2, 0, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      1}, /* memwr_dst */

     {0, /* layer_type */
      32, 32, 64, 128, 64, 3, 3, 128, /* memRd params */
      1, /* memrd_src */
      16, 16, 128, 2, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      2}, /* memwr_dst */
     {0, /* layer_type */
      16, 16, 128, 128, 128, 3, 3, 128, /* memRd params */
      2, /* memrd_src */
      16, 16, 128, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2, /* normalization */
      1, /* shortcut */
      0}, /* memwr_dst */

	  /* Basic Block 1 */
     {0, /* layer_type */
      16, 16, 128, 128, 128, 3, 3, 128, /* memRd params */
      0, /* memrd_src */
      16, 16, 128, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      1}, /* memwr_dst */
     {0, /* layer_type */
      16, 16, 128, 128, 128, 3, 3, 128, /* memRd params */
      1, /* memrd_src */
      16, 16, 128, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2, /* normalization */
      0, /* shortcut */
      2}, /* memwr_dst */

      /* layer 3 */
	  /* Basic Block 0 */

	 /* shortcut */
	 {0, /* layer_type */
      16, 16, 128, 256, 128, 1, 1, 256, /* memRd params */
      2, /* memrd_src */
      8, 8, 256, 2, 0, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      0}, /* memwr_dst */

     {0, /* layer_type */
      16, 16, 128, 256, 128, 3, 3, 256, /* memRd params */
      2, /* memrd_src */
      8, 8, 256, 2, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      1}, /* memwr_dst */
     {0, /* layer_type */
      8, 8, 256, 256, 256, 3, 3, 256, /* memRd params */
      1, /* memrd_src */
      8, 8, 256, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2, /* normalization */
      0, /* shortcut */
      2}, /* memwr_dst */

	  /* Basic Block 1 */
     {0, /* layer_type */
      8, 8, 256, 256, 256, 3, 3, 256, /* memRd params */
      2, /* memrd_src */
      8, 8, 256, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      1}, /* memwr_dst */
     {0, /* layer_type */
      8, 8, 256, 256, 256, 3, 3, 256, /* memRd params */
      1, /* memrd_src */
      8, 8, 256, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2, /* normalization */
      2, /* shortcut */
      0}, /* memwr_dst */

      /* layer 4 */
	  /* Basic Block 0 */

	 /* shortcut */
	 {0, /* layer_type */
      8, 8, 256, 512, 256, 1, 1, 512, /* memRd params */
      0, /* memrd_src */
      4, 4, 512, 2, 0, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      1}, /* memwr_dst */

     {0, /* layer_type */
      8, 8, 256, 512, 256, 1, 1, 512, /* memRd params */
      0, /* memrd_src */
      4, 4, 512, 2, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      2}, /* memwr_dst */
     {0, /* layer_type */
      4, 4, 512, 512, 512, 3, 3, 512, /* memRd params */
      2, /* memrd_src */
      4, 4, 512, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2, /* normalization */
      1, /* shortcut */
      0}, /* memwr_dst */

	  /* Basic Block 1 */
     {0, /* layer_type */
      4, 4, 512, 512, 512, 3, 3, 512, /* memRd params */
      0, /* memrd_src */
      4, 4, 512, 1, 1, 0, 1, /* conv params */
      0, 13, 13, 256, 3, 2, /* pooling */
      2,  /* normalization */
      4, /* shortcut */
      2}, /* memwr_dst */
     {0, /* layer_type */
      4, 4, 512, 512, 512, 3, 3, 512, /* memRd params */
      2, /* memrd_src */
      4, 4, 512, 1, 1, 0, 1, /* conv params */
      2, 4, 4, 512, 4, 4, /* pooling */
      2, /* normalization */
      0, /* shortcut */
      3}, /* memwr_dst */


	/* fc layer */ 
	  {1, /* layer_type */
      4, 4, 512, 10, 512, 0, 0, 10, /* memRd params */
      3, /* memrd_src */
      4, 4, 512, 1, 1, 0, 1, /* conv params */
      0, 4, 4, 512, 4, 4, /* pooling */
      0, /* normalization */
      4, /* shortcut */
      4} /* memwr_dst */
    };

char precision_config[][3] ={{8,  0, -4},//Layer-1
                             {8,  0, -2},//Layer-2
};
unsigned input_config[5] = {227, 227, 3, 1};  //original image size(dim1, dim2, dim3), batch size
unsigned output_config[3] = {55, 55, 96}; //Layer-8  Note: only one result is extracted and verified


/*
// Test with batch=1
// VGG-16 Configuration
unsigned layer_config[][NUM_CONFIG_ITEM] = {{0,
							224, 224, 3, 3, 3, 3, 64, 64,
							0,
							224, 224, 64, 1, 1, 0, 1,
							0, 224, 224, 64, 0, 0,
							0,
							1},//Layer-1 (conv1_1)
							{0,
							224, 224, 64, 3, 3, 64, 64, 64,
							1,
							224, 224, 64, 1, 1, 0, 1,
							1, 112, 112, 64, 2, 2,
							0,
							0},//Layer-2 (conv1_2)
							{0,
							112, 112, 64, 3, 3, 64, 128, 128,
							0,
							112, 112, 128, 1, 1, 0, 1,
							0, 112, 112, 128, 0, 0,
							0,
							1},//Layer-3 (conv2_1)
							{0,
							112, 112, 128, 3, 3, 128, 128, 128,
							1,
							112, 112, 128, 1, 1, 0, 1,
							1, 56, 56, 128, 2, 2,
							0,
							0},//Layer-4 (conv2_2)
							{0,
							56, 56, 128, 3, 3, 128, 256, 256,
							0,
							56, 56, 256, 1, 1, 0, 1,
							0, 56, 56, 256, 0, 0,
							0,
							1},//Layer-5 (conv3_1)
							{0,
							56, 56, 256, 3, 3, 256, 256, 256,
							1,
							56, 56, 256, 1, 1, 0, 1,
							0, 56, 56, 256, 0, 0,
							0,
							0},//Layer-6 (conv3_2)
							{0,
							56, 56, 256, 3, 3, 256, 256, 256,
							0,
							56, 56, 256, 1, 1, 0, 1,
							1, 28, 28, 256, 2, 2,
							0,
							1},//Layer-7 (conv3_3)
							{0,
							28, 28, 256, 3, 3, 256, 512, 512,
							1,
							28, 28, 512, 1, 1, 0, 1,
							0, 28, 28, 512, 0, 0,
							0,
							0},//Layer-8  (conv4_1)
							{0,
							28, 28, 512, 3, 3, 512, 512, 512,
							0,
							28, 28, 512, 1, 1, 0, 1,
							0, 28, 28, 512, 0, 0,
							0,
							1},//Layer-9  (conv4_2)
							{0,
							28, 28, 512, 3, 3, 512, 512, 512,
							1,
							28, 28, 512, 1, 1, 0, 1,
							1, 14, 14, 512, 2, 2,
							0,
							0},//Layer-10 (conv4_3)
							{0,
							14, 14, 512, 3, 3, 512, 512, 512,
							0,
							14, 14, 512, 1, 1, 0, 1,
							0, 14, 14, 512, 0, 0,
							0,
							1},//Layer-11  (conv5_1)
							{0,
							14, 14, 512, 3, 3, 512, 512, 512,
							1,
							14, 14, 512, 1, 1, 0, 1,
							0, 14, 14, 512, 0, 0,
							0,
							0},//Layer-12  (conv5_2)
							{0,
							14, 14, 512, 3, 3, 512, 512, 512,
							0,
							14, 14, 512, 1, 1, 0, 1,
							1, 7, 7, 512, 2, 2,
							0,
							2},//Layer-13  (conv5_3)    Note: for last conv layer, outputs are write to fc buffer
							{1,
							7, 7, 512, 7, 7, 512, 4096, 4096,
							2,
							1, 1, 4096, 7, 0, 0, 1,
							0, 1, 1, 4096, 0, 0,
							0,
							3},//Layer-14  (fc6)							
							{1,
							1, 1, 4096, 1, 1, 4096, 4096, 4096,
							3,
							1, 1, 4096, 1, 0, 0, 1,
							0, 1, 1, 4096, 0, 0,
							0,
							2},//Layer-15  (fc7)
							{1,
							1, 1, 4096, 1, 1, 4096, 1024, 1024,
							2,
							1, 1, 1024, 1, 0, 0, 0,
							0, 1, 1, 1024, 0, 0,
							0,
							3}//Layer-16  (fc8)		
							};

char precision_config[][3] ={{7,  0, -2},//Layer-1
							{ 8, -2, -5},//Layer-2
							{ 8, -5, -5},//Layer-3
							{ 8, -5, -6},//Layer-4
							{ 7, -6, -7},//Layer-5
							{ 8, -7, -7},//Layer-6
							{ 8, -7, -7},//Layer-7
							{ 8, -7, -6},//Layer-8
							{ 8, -6, -5},//Layer-9
							{ 8, -5, -5},//Layer-10
							{ 9, -5, -4},//Layer-11
							{ 9, -4, -3},//Layer-12
							{ 8, -3, -2},//Layer-13
							{ 8, -2,  0},//Layer-14
							{ 7,  0,  2},//Layer-15
							{ 7,  2,  2}//Layer-16
							};

unsigned input_config[4] = {224, 224, 3, 1};

//unsigned output_config[3] = {224, 224, 64};//Layer-1

//unsigned output_config[3] = {56, 56, 128};//Layer-4(pool2)

//unsigned output_config[3] = {28, 28, 256};//Layer-7(pool3)

//unsigned output_config[3] = {28, 28, 512};//Layer-8(relu4_1)

//unsigned output_config[3] = {28, 28, 512};//Layer-9(relu4_2)

//unsigned output_config[3] = {14, 14, 512};//Layer-10(pool4)

//unsigned output_config[3] = {7, 7, 512};//Layer-13(pool5)

//unsigned output_config[3] = {1, 1, 4096};//Layer-14

unsigned output_config[3] = {1, 1, 1024};//Layer-16
*/
	

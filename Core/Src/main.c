/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

COM_InitTypeDef BspCOMInit;

SPI_HandleTypeDef hspi1;

TIM_HandleTypeDef htim2;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
	// Registre LSM6DSR
	#define LSM6DSR_WHO_AM_I        0x0F
	#define LSM6DSR_CTRL1_XL        0x10  // Accelerometru config
	#define LSM6DSR_CTRL2_G         0x11  // Giroscop config
	#define LSM6DSR_OUTX_L_G        0x22  // Inceput date giroscop
	#define LSM6DSR_OUTX_L_A        0x28  // Inceput date accelerometru
	#define LSM6DSR_WHO_AM_I_VAL    0x6C  // Valoarea asteptata

	#define NUM_SENSORS 5
	#define READ_FLAG   0x80  // Bitul 7 pus pe  1 pentru citire SPI

	// Pinii CS pentru fiecare senzor
	GPIO_TypeDef* CS_PORT[NUM_SENSORS] = {GPIOC, GPIOC, GPIOA, GPIOF, GPIOE};
	uint16_t      CS_PIN[NUM_SENSORS]  = {GPIO_PIN_11, GPIO_PIN_6, GPIO_PIN_0, GPIO_PIN_10, GPIO_PIN_12};

	typedef struct {
		int16_t gx, gy, gz;        // Giroscop brut
		int16_t ax, ay, az;        // Accelerometru brut
		float gx_dps, gy_dps, gz_dps;  // Giroscop in grade/secunda
		float ax_g, ay_g, az_g;        // Accelerometru in g
		uint32_t timestamp_us;
	} IMU_Data;

	IMU_Data imu[NUM_SENSORS];


	// pentru dataset
	typedef struct __attribute__((packed)) {
	    uint8_t  header[4];        // 4 bytes header
	    uint32_t packet_id;
	    uint32_t timestamp_us[NUM_SENSORS];
	    float    gx[NUM_SENSORS], gy[NUM_SENSORS], gz[NUM_SENSORS];
	    float    ax[NUM_SENSORS], ay[NUM_SENSORS], az[NUM_SENSORS];
	    uint8_t  checksum;
	} UARTPacket;

	UARTPacket uart_packet;
	uint32_t packet_id = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_SPI1_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
	void CS_Low(uint8_t sensor) {
		HAL_GPIO_WritePin(CS_PORT[sensor], CS_PIN[sensor], GPIO_PIN_RESET);
	}

	void CS_High(uint8_t sensor) {
		HAL_GPIO_WritePin(CS_PORT[sensor], CS_PIN[sensor], GPIO_PIN_SET);
	}

	void IMU_WriteReg(uint8_t sensor, uint8_t reg, uint8_t value) {
		uint8_t tx[2] = {reg & 0x7F, value};  // Bitul 7 pus pe 0 pentru scriere
		CS_Low(sensor);
		HAL_SPI_Transmit(&hspi1, tx, 2, HAL_MAX_DELAY);
		CS_High(sensor);
	}

	uint8_t IMU_ReadReg(uint8_t sensor, uint8_t reg) {
		uint8_t tx = reg | READ_FLAG;
		uint8_t rx = 0;
		CS_Low(sensor);
		HAL_SPI_Transmit(&hspi1, &tx, 1, HAL_MAX_DELAY);
		HAL_SPI_Receive(&hspi1, &rx, 1, HAL_MAX_DELAY);
		CS_High(sensor);
		return rx;
	}

	void IMU_ReadRegs(uint8_t sensor, uint8_t reg, uint8_t *buf, uint8_t len) {
		uint8_t tx = reg | READ_FLAG;
		CS_Low(sensor);
		HAL_SPI_Transmit(&hspi1, &tx, 1, HAL_MAX_DELAY);
		HAL_SPI_Receive(&hspi1, buf, len, HAL_MAX_DELAY);
		CS_High(sensor);
	}

	uint8_t IMU_Init(uint8_t sensor) {
		uint8_t who_am_i = IMU_ReadReg(sensor, LSM6DSR_WHO_AM_I);
		if (who_am_i != LSM6DSR_WHO_AM_I_VAL) {
			//printf("Senzor %d: WHO_AM_I gresit: 0x%02X\r\n", sensor + 1, who_am_i);
			return 0;
		}

		// Accelerometru: 104 Hz, +-4g
		IMU_WriteReg(sensor, LSM6DSR_CTRL1_XL, 0x48);
		// Giroscop: 104 Hz, +-500 dps
		IMU_WriteReg(sensor, LSM6DSR_CTRL2_G, 0x44);

		//printf("Senzor %d: initializat OK (WHO_AM_I=0x%02X)\r\n", sensor + 1, who_am_i);
		return 1;
	}

	void IMU_ReadData(uint8_t sensor) {
		uint8_t buf[12];

		imu[sensor].timestamp_us = __HAL_TIM_GET_COUNTER(&htim2);

		// Citeste giroscop (6 bytes) si accelerometru (6 bytes)
		IMU_ReadRegs(sensor, LSM6DSR_OUTX_L_G, buf, 12);

		imu[sensor].gx = (int16_t)(buf[1]  << 8 | buf[0]);
		imu[sensor].gy = (int16_t)(buf[3]  << 8 | buf[2]);
		imu[sensor].gz = (int16_t)(buf[5]  << 8 | buf[4]);
		imu[sensor].ax = (int16_t)(buf[7]  << 8 | buf[6]);
		imu[sensor].ay = (int16_t)(buf[9]  << 8 | buf[8]);
		imu[sensor].az = (int16_t)(buf[11] << 8 | buf[10]);

		// 500dps range cred ca e 17.5 mdps/LSB si tre sa impart la 1000 sa ajung la g
		float gyro_sens = 17.5f / 1000.0f;
		imu[sensor].gx_dps = imu[sensor].gx * gyro_sens;
		imu[sensor].gy_dps = imu[sensor].gy * gyro_sens;
		imu[sensor].gz_dps = imu[sensor].gz * gyro_sens;

		// 4g range cred ca e 0.122 mg/LSB, si tre sa impart la 1000 sa ajung la g
		float acc_sens = 0.122f / 1000.0f;
		imu[sensor].ax_g = imu[sensor].ax * acc_sens;
		imu[sensor].ay_g = imu[sensor].ay * acc_sens;
		imu[sensor].az_g = imu[sensor].az * acc_sens;
	}

	void IMU_PrintData(uint8_t sensor) {
		printf("Senzor %d | t=%lu us \r\n",
			   sensor + 1, imu[sensor].timestamp_us);
		printf("Gyro [dps]: X=%7.3f  Y=%7.3f  Z=%7.3f\r\n",
			   imu[sensor].gx_dps, imu[sensor].gy_dps, imu[sensor].gz_dps);
		printf("Accel [g]: X=%7.4f  Y=%7.4f  Z=%7.4f\r\n",
			   imu[sensor].ax_g, imu[sensor].ay_g, imu[sensor].az_g);
	}

	// pentru dataset

	uint8_t compute_checksum(uint8_t *data, uint16_t len) {
	    uint8_t cs = 0;
	    for (uint16_t i = 0; i < len; i++) cs ^= data[i];
	    return cs;
	}

	void UART_SendPacket(void) {
		uart_packet.header[0] = 0xAA;
		uart_packet.header[1] = 0xBB;
		uart_packet.header[2] = 0xCC;
		uart_packet.header[3] = 0xDD;
	    uart_packet.packet_id = packet_id++;

	    for (int i = 0; i < NUM_SENSORS; i++) {
	        uart_packet.timestamp_us[i] = imu[i].timestamp_us;
	        uart_packet.gx[i] = imu[i].gx_dps;
	        uart_packet.gy[i] = imu[i].gy_dps;
	        uart_packet.gz[i] = imu[i].gz_dps;
	        uart_packet.ax[i] = imu[i].ax_g;
	        uart_packet.ay[i] = imu[i].ay_g;
	        uart_packet.az[i] = imu[i].az_g;
	    }

	    uart_packet.checksum = compute_checksum(
	        (uint8_t*)&uart_packet,
	        sizeof(UARTPacket) - 1
	    );

	    HAL_UART_Transmit(&huart2,
	        (uint8_t*)&uart_packet,
	        sizeof(UARTPacket),
	        HAL_MAX_DELAY
	    );
	}


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_SPI1_Init();
  MX_TIM2_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
	  HAL_TIM_Base_Start(&htim2);
	  // Toti CS ridicati la inceput (idle HIGH)
	  for (int i = 0; i < NUM_SENSORS; i++) CS_High(i);
	  HAL_Delay(10);

	  // Initializare senzori
	  uint8_t sensor_ok[NUM_SENSORS];
	  for (int i = 0; i < NUM_SENSORS; i++) {
		  sensor_ok[i] = IMU_Init(i);
	  }
	  uint8_t size_check = sizeof(UARTPacket);
	  HAL_Delay(2000);
  /* USER CODE END 2 */

  /* Initialize leds */
  BSP_LED_Init(LED_GREEN);
  BSP_LED_Init(LED_BLUE);
  BSP_LED_Init(LED_RED);

  /* Initialize USER push-button, will be used to trigger an interrupt each time it's pressed.*/
  BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);

  /* Initialize COM1 port (115200, 8 bits (7-bit data + 1 stop bit), no parity */
  BspCOMInit.BaudRate   = 115200;
  BspCOMInit.WordLength = COM_WORDLENGTH_8B;
  BspCOMInit.StopBits   = COM_STOPBITS_1;
  BspCOMInit.Parity     = COM_PARITY_NONE;
  BspCOMInit.HwFlowCtl  = COM_HWCONTROL_NONE;
  if (BSP_COM_Init(COM1, &BspCOMInit) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	  while (1)
	  {
		  for (int i = 0; i < NUM_SENSORS; i++) {
		      if (sensor_ok[i]) {
		          IMU_ReadData(i);
		          //IMU_PrintData(i);
		      }
		  }
		  UART_SendPacket();
		  //printf("\r\n");
		  HAL_Delay(50);
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOMEDIUM;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_32;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 0x0;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  hspi1.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
  hspi1.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
  hspi1.Init.TxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi1.Init.RxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi1.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
  hspi1.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
  hspi1.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
  hspi1.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
  hspi1.Init.IOSwap = SPI_IO_SWAP_DISABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 79;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 4294967295;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart2, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart2, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOF, GPIO_PIN_10, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_12, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_6|GPIO_PIN_11, GPIO_PIN_SET);

  /*Configure GPIO pin : PF10 */
  GPIO_InitStruct.Pin = GPIO_PIN_10;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);

  /*Configure GPIO pin : PA0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : PE12 */
  GPIO_InitStruct.Pin = GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pins : PC6 PC11 */
  GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_11;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*AnalogSwitch Config */
  HAL_SYSCFG_AnalogSwitchConfig(SYSCFG_SWITCH_PA0, SYSCFG_SWITCH_PA0_CLOSE);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
int _write(int file, char *ptr, int len)
{
	HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, HAL_MAX_DELAY);
	for (int i = 0; i < len; i++) ITM_SendChar(ptr[i]);
	    return len;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

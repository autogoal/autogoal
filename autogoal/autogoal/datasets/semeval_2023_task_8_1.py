import json
import numpy as np
import os
from autogoal.datasets import datapath, download
import csv
import re
import enum

class TaskTypeSemeval(enum.Enum):
    TokenClassification="TokenClassification",
    SentenceClassification="SentenceClassification",
    SentenceMultilabelClassification="SentenceMultilabelClassification"
    
class SemevalDatasetSelection(enum.Enum):
    Original="Original",
    Actual="Actual",

class TargetClassesMapping(enum.Enum):
    Original="Original",
    Extended="Extended",
    
class TestSplitSize(enum.Enum):
    Thirty="Thirty",
    Twenty="Twenty",
    
TEST_SPLIT_SIZE_TO_INDEXES = {
    TestSplitSize.Thirty: [683, 4712, 4352, 2430, 1865, 1266, 3395, 246, 4307, 3688, 2289, 4137, 2440, 2300, 4434, 4988, 2920, 1810, 4265, 1096, 2256, 1111, 4558, 2104, 1115, 1147, 4904, 4032, 1609, 3499, 3607, 4026, 1151, 1872, 3688, 3630, 899, 307, 1840, 2610, 2735, 5089, 576, 5236, 398, 4342, 1591, 1483, 3473, 634, 1700, 3627, 2323, 528, 1792, 4946, 3180, 947, 2039, 3370, 428, 4657, 2498, 1828, 4898, 3267, 1627, 1001, 2557, 2212, 2514, 2160, 2233, 2908, 1729, 2647, 1153, 1249, 1556, 5147, 3457, 464, 3389, 1470, 1068, 3483, 1326, 3147, 3500, 261, 2376, 5048, 3464, 3921, 3437, 1915, 3461, 685, 1514, 1278, 4558, 3384, 135, 3894, 3742, 1271, 421, 4480, 5211, 2037, 1663, 4052, 2979, 2324, 1769, 2070, 2799, 1219, 4881, 4104, 2099, 5026, 872, 1182, 2715, 3410, 743, 3373, 3650, 4075, 871, 4125, 1858, 4201, 2693, 4437, 581, 1054, 947, 138, 2682, 4131, 722, 35, 826, 2503, 2740, 2812, 1751, 4843, 872, 245, 531, 2987, 4381, 1689, 2387, 4552, 2764, 4876, 113, 1958, 4966, 786, 2676, 3129, 598, 4053, 2358, 2729, 3248, 3389, 4746, 324, 994, 1697, 4975, 1014, 501, 4342, 4351, 4764, 2788, 5084, 5223, 3243, 2163, 5101, 2818, 4005, 1314, 4999, 647, 4910, 3895, 3395, 231, 881, 1194, 1859, 2594, 5042, 3799, 1853, 1483, 2116, 4551, 1926, 4670, 4633, 3289, 0, 4578, 1912, 102, 462, 647, 4951, 1317, 4875, 2491, 5039, 2929, 3767, 3447, 5012, 4805, 1646, 1762, 1751, 939, 688, 3513, 3153, 1702, 5159, 2377, 29, 3608, 2461, 2927, 5233, 4629, 61, 1027, 769, 472, 3737, 1595, 1744, 5192, 3898, 4075, 1149, 1947, 3090, 3840, 661, 2163, 42, 2546, 4693, 3107, 2851, 4149, 83, 4065, 719, 4904, 1297, 1040, 465, 2249, 308, 190, 172, 3676, 1692, 97, 4085, 2163, 31, 474, 3968, 4646, 480, 3054, 3248, 5002, 674, 1582, 1658, 3068, 2198, 3121, 5056, 1744, 2202, 5209, 33, 5108, 1804, 650, 1944, 226, 2086, 4050, 5169, 1679, 2091, 1441, 1847, 1474, 2940, 5252, 763, 1407, 4844, 4022, 4107, 4768, 1966, 3397, 2465, 2897, 2867, 3267, 3554, 4969, 1687, 767, 5205, 933, 4976, 1333, 617, 3725, 2775, 1498, 4201, 216, 796, 3334, 1408, 4467, 959, 729, 3952, 2397, 2269, 3904, 1457, 459, 971, 3636, 3219, 1481, 3935, 3649, 1011, 3536, 3374, 3507, 2439, 960, 2531, 2752, 3599, 3679, 2497, 1297, 3398, 5016, 5096, 3357, 4519, 411, 3750, 232, 3779, 4498, 469, 4609, 3055, 4024, 2000, 4936, 4962, 1622, 4409, 49, 4312, 3885, 2018, 4712, 2295, 2379, 4846, 284, 4718, 309, 4814, 4413, 1160, 4482, 3260, 3364, 628, 1544, 4206, 3825, 26, 1387, 3364, 1890, 1077, 963, 3613, 2695, 1549, 333, 437, 4928, 3700, 1136, 1691, 4218, 3798, 1371, 2099, 4849, 3785, 3835, 4829, 1147, 3559, 2613, 4221, 571, 4761, 779, 862, 3716, 4190, 2492, 3355, 4988, 2092, 4314, 4073, 3332, 1532, 2283, 4084, 2603, 5035, 990, 2816, 3799, 2940, 3319, 2217, 4972, 2688, 3220, 1696, 2531, 1640, 1093, 4427, 203, 3363, 2585, 3902, 55, 4126, 2211, 1935, 1978, 3787, 1728, 2782, 4476, 5256, 2536, 4115, 4354, 2626, 4382, 2560, 4415, 3710, 2535, 4112, 4296, 4026, 475, 1123, 4114, 631, 1516, 1030, 780, 2982, 4268, 1925, 511, 1058, 1365, 3013, 5196, 3758, 4476, 4546, 3136, 7, 5055, 2952, 2711, 2427, 3073, 5215, 5085, 2501, 1978, 3404, 5218, 2066, 2817, 3148, 1520, 781, 1898, 1884, 4280, 4504, 3428, 5124, 4831, 145, 4127, 2178, 4760, 4066, 4354, 4683, 59, 376, 3522, 5037, 5112, 2936, 171, 399, 4117, 4570, 3031, 3577, 1455, 3496, 1852, 1757, 1146, 2082, 4771, 1840, 664, 2298, 2602, 655, 3369, 12, 4734, 3881, 96, 3182, 3385, 2470, 2338, 4523, 3799, 120, 5016, 2165, 3160, 3609, 3835, 2371, 1603, 4857, 3169, 3724, 760, 2518, 407, 43, 2213, 2393, 594, 3413, 1706, 3625, 935, 4042, 656, 599, 2277, 3879, 4585, 1092, 897, 4631, 2788, 2816, 35, 1154, 4021, 497, 1532, 3329, 3176, 2390, 4965, 2161, 4368, 3135, 4599, 140, 585, 1288, 1773, 3069, 1842, 5100, 1050, 2925, 3167, 3395, 2667, 1204, 4604, 4773, 3612, 4099, 4925, 5087, 4322, 3155, 4933, 4558, 4603, 2949, 5147, 670, 799, 4484, 3247, 3163, 45, 1263, 4007, 135, 2936, 1407, 2679, 2622, 1256, 1329, 1927, 1282, 274, 4555, 5208, 2664, 789, 881, 887, 3756, 4012, 4936, 1666, 2857, 4264, 2964, 1099, 1610, 3128, 3559, 3421, 1231, 614, 3323, 1673, 2851, 2912, 2830, 1268, 4138, 275, 5191, 2373, 3914, 5031, 4023, 1219, 4372, 3783, 4087, 4571, 2663, 5152, 590, 5220, 3636, 2942, 2750, 149, 2564, 2730, 2996, 789, 2930, 1168, 419, 47, 2966, 3857, 2362, 124, 54, 1292, 3925, 745, 1295, 4601, 2890, 1433, 1586, 4275, 2504, 1121, 3875, 1652, 4552, 949, 578, 2648, 4934, 109, 5233, 1971, 5269, 4076, 1359, 4170, 4136, 1675, 1432, 5200, 1878, 3675, 3901, 2995, 1071, 4490, 3196, 1384, 181, 2842, 4530, 4202, 1150, 2663, 3455, 4059, 4657, 2459, 3889, 2214, 1402, 5198, 5125, 2915, 3746, 746, 683, 4518, 1013, 4266, 3028, 2619, 4904, 2437, 1907, 94, 3067, 3986, 4019, 3731, 2055, 3939, 5172, 3829, 4054, 1644, 3946, 4500, 2237, 4662, 685, 2365, 4313, 2225, 1198, 1329, 2517, 4989, 2525, 663, 3299, 2029, 2832, 1574, 1507, 4720, 5200, 4390, 2755, 2567, 1966, 4363, 2872, 5119, 142, 1069, 1768, 2404, 3030, 4317, 1573, 935, 3016, 4609, 4216, 2215, 3921, 5250, 3264, 4505, 3137, 1262, 1525, 2614, 3871, 717, 740, 1430, 2489, 1735, 1562, 2746, 1245, 4926, 614, 1873, 4195, 195, 1922, 5126, 323, 2702, 2583, 500, 1891, 4102, 3024, 4270, 3032, 3612, 2375, 268, 1725, 3866, 3872, 3957, 1188, 713, 1112, 2229, 2311, 4032, 917, 4637, 2819, 1226, 2170, 3320, 201, 4714, 4403, 2984, 3127, 4456, 3794, 3426, 917, 2992, 4644, 371, 4115, 4791, 759, 3247, 3000, 3400, 4028, 2347, 1995, 4431, 4030, 2537, 2860, 5140, 3339, 2945, 1246, 1143, 4265, 642, 3848, 634, 4081, 508, 3388, 378, 1244, 4093, 2858, 5109, 615, 4327, 4761, 4030, 1722, 2608, 997, 1460, 1643, 3985, 1484, 2831, 2620, 3699, 4762, 2485, 2523, 4430, 2945, 3488, 495, 1295, 1981, 1859, 2531, 301, 3787, 1994, 5025, 1723, 1569, 2818, 4523, 373, 2351, 3434, 1343, 4876, 3302, 2597, 5124, 1518, 2901, 739, 4549, 3536, 1333, 1609, 4546, 2518, 3830, 515, 5207, 2156, 2473, 3617, 3594, 350, 2095, 4116, 2516, 1926, 1894, 572, 2918, 4379, 5173, 2222, 1006, 1803, 4099, 703, 2575, 3057, 3615, 4739, 2887, 3497, 1767, 2523, 1927, 2346, 1620, 4027, 2200, 408, 3855, 4807, 4029, 3752, 1354, 1416, 4237, 2453, 1998, 793, 2002, 3951, 2790, 353, 1860, 2503, 1140, 726, 244, 4837, 1736, 2187, 2333, 4394, 4259, 476, 1267, 4808, 2266, 4041, 4392, 2473, 1570, 3075, 1515, 2801, 1253, 2327, 605, 4764, 1004, 327, 3834, 3395, 4524, 1613, 2005, 1787, 5022, 4687, 3266, 291, 3254, 1285, 5121, 3460, 131, 4546, 476, 5183, 901, 1481, 259, 782, 313, 3883, 3940, 1615, 5029, 361, 3963, 108, 4096, 3609, 5116, 1982, 4899, 4690, 958, 3240, 1245, 4602, 1163, 3752, 4340, 4710, 2376, 1262, 2609, 3218, 2794, 720, 2643, 1309, 1571, 1590, 1813, 5212, 4642, 3717, 445, 510, 3537, 1373, 1114, 162, 2279, 369, 1712, 3243, 2551, 4084, 2296, 3645, 318, 1904, 768, 2621, 2417, 3571, 1718, 3599, 1913, 2406, 5163, 4415, 4369, 3817, 4121, 5194, 2153, 1905, 2907, 640, 1963, 3533, 3426, 4844, 4177, 971, 1712, 5241, 4468, 2844, 2395, 1225, 4018, 525, 6, 1554, 1321, 3133, 3899, 5187, 5256, 1152, 3363, 3067, 895, 4351, 3229, 2286, 3985, 946, 4907, 1633, 1308, 3019, 4142, 3092, 1884, 4183, 428, 41, 3255, 2839, 3991, 4461, 4319, 4331, 3739, 661, 4210, 2059, 3288, 1786, 4366, 442, 1140, 3925, 3635, 3731, 971, 4758, 78, 3564, 629, 4219, 2273, 2780, 3607, 4571, 1367, 523, 1859, 3104, 805, 3284, 3924, 4951, 1837, 1808, 4952, 1856, 3822, 3496, 222, 4464, 2929, 4770, 4931, 3121, 3603, 4055, 5017, 2214, 788, 4491, 2425, 3590, 2840, 4513, 2075, 4032, 3392, 94, 2145, 2761, 1837, 3557, 1687, 1073, 4043, 5227, 814, 4324, 211, 545, 4308, 3952, 954, 4571, 5065, 4539, 27, 3098, 1168, 3350, 658, 4734, 2410, 101, 2262, 1276, 2363, 432, 3130, 2498, 2121, 2224, 748, 2461, 482, 1000, 5245, 771, 2321, 2622, 4375, 3725, 4376, 1392, 3102, 4809, 4958, 420, 3489, 3616, 1838, 2717, 485, 4734, 4114, 1578, 2897, 1012, 3730, 4132, 1290, 3300, 2981, 1896, 2711, 4480, 1218, 11, 896, 580, 3703, 870, 1887, 3419, 3787, 904, 795, 147, 3763, 1049, 1773, 1373, 1885, 2466, 3846, 4215, 1440, 3502, 4149, 3716, 1156, 5069, 4226, 2106, 4075, 4524, 298, 4324, 453, 2063, 16, 3064, 4658, 4567, 543, 601, 2624, 710, 1191, 2951, 1682, 436, 961, 3427, 4981, 4704, 2533, 1557, 4686, 3965, 3218, 4341, 3117, 2864, 3386, 3126, 1403, 3457, 5031, 4801, 693, 2435, 2342, 3084, 2553, 2161, 5128, 4037, 1068, 3946, 361, 220, 117, 3160, 372, 280, 3290, 5200, 5151, 4084, 138, 3091, 3729, 4174, 2253, 4863, 548, 4733, 1906, 1429, 1407, 3058, 2668, 2716, 2945, 1533, 2129, 3771, 4064, 4301, 4607, 3496, 1788, 2279, 163, 2209, 4874, 2788, 702, 4955, 4451, 2596, 581, 1208, 495, 102, 4801, 1714, 2147, 4400, 4485, 1442, 3368, 1316, 5169, 4358, 1412, 2215, 5116, 4215, 531, 3315, 1851, 1583, 3910, 5009, 4296, 244, 453, 4850, 1776, 607, 3899, 2720, 438, 4602, 1792, 4484, 1561, 3121, 2387, 2002, 5217, 802, 2208, 4782, 2451, 1366, 637, 4773, 3335, 706, 748, 2049, 1629, 1784, 1476, 4240, 4930, 475, 4955, 4472, 1339, 3878, 4793, 692, 5063, 870, 2100, 4753, 234, 1491, 1665, 2410, 5056, 2667, 4169, 1347, 2924, 3143, 145, 2694, 2610, 5186, 1953, 935, 3328, 389, 1267, 2712, 2395, 2033, 2104, 5242, 4061, 1528, 4904, 2059, 3702, 2858, 4026, 2742, 1715, 802, 4158, 4562, 547, 3843, 4458, 3513, 461, 24, 115, 1611, 199, 1665, 3073, 4692, 1725, 2878, 195, 1960, 4679, 1361, 4650, 2830, 1348, 3491, 1794, 3215, 1868, 4780, 170, 5077, 3687, 4781, 4609, 4494],
    TestSplitSize.Twenty: [5180, 403, 1310, 4853, 4231, 159, 4286, 1893, 483, 237, 5133, 1634, 4323, 2428, 4223, 267, 469, 1010, 80, 1045, 246, 2379, 3824, 654, 2612, 396, 1519, 905, 796, 2494, 2226, 2090, 154, 514, 5016, 3548, 2245, 4357, 1339, 657, 2266, 708, 153, 5087, 2112, 3566, 493, 102, 2394, 2579, 2382, 3659, 2905, 4961, 4715, 4325, 1843, 2607, 218, 1825, 4666, 3306, 4220, 4326, 636, 3450, 4707, 3818, 1597, 2541, 3141, 259, 1703, 5191, 3747, 4106, 467, 3480, 538, 2573, 1849, 2916, 509, 5031, 4607, 103, 4860, 1284, 35, 197, 3880, 4841, 152, 3838, 4560, 3353, 2815, 4088, 3621, 1941, 830, 628, 1654, 737, 2482, 420, 1060, 1021, 4464, 562, 3864, 4074, 4735, 4425, 1136, 2260, 4507, 1867, 5020, 3171, 429, 438, 4142, 5152, 614, 4325, 1008, 255, 3362, 5198, 863, 1097, 4146, 1660, 4156, 190, 3175, 3562, 4329, 3387, 1282, 3421, 3504, 442, 3598, 634, 256, 4290, 3924, 3685, 2602, 1037, 76, 1384, 3798, 4436, 4233, 4879, 1266, 1700, 2184, 83, 4946, 633, 590, 401, 4470, 4526, 3540, 5196, 4006, 3160, 4149, 1228, 34, 460, 1329, 3730, 745, 3501, 1438, 1540, 3557, 3451, 2571, 5026, 3940, 3090, 1881, 3692, 1278, 986, 2598, 4375, 1380, 4247, 452, 4766, 4309, 3524, 1255, 4246, 1717, 1898, 752, 3779, 2227, 882, 4395, 3187, 1723, 4497, 1672, 4875, 3143, 2187, 3009, 2591, 2614, 599, 2414, 5071, 4712, 524, 2657, 826, 1572, 4979, 3340, 4105, 4680, 4354, 4796, 4671, 4107, 227, 3713, 4717, 2138, 4376, 2183, 5187, 1624, 1430, 4024, 508, 367, 2097, 2036, 1184, 3793, 171, 4663, 3771, 2940, 3942, 2076, 3759, 4465, 4093, 1303, 2339, 970, 4271, 4684, 3795, 5030, 713, 3379, 5005, 220, 4125, 3447, 2068, 2887, 884, 1142, 5243, 4683, 1343, 3913, 3360, 2624, 4067, 4028, 2704, 286, 2782, 5149, 5186, 3690, 1578, 192, 4639, 67, 3193, 1212, 225, 4203, 3731, 736, 3199, 1075, 3264, 3922, 3399, 4501, 2590, 3035, 3835, 5206, 5240, 608, 4749, 5132, 1580, 3743, 2748, 707, 3343, 5268, 2009, 571, 1306, 899, 1208, 666, 2337, 3315, 1530, 4311, 372, 321, 2981, 2835, 3411, 1497, 1606, 4815, 3004, 603, 2568, 2117, 4386, 3017, 3901, 3007, 4212, 1472, 3634, 162, 3060, 1153, 4618, 2786, 502, 1375, 4995, 4454, 3489, 1747, 712, 2660, 2645, 1884, 4673, 1722, 1122, 3604, 4488, 2585, 1890, 3585, 1552, 1157, 924, 4976, 1175, 542, 5050, 3422, 176, 5159, 1617, 2927, 5222, 4988, 582, 2694, 2825, 2512, 997, 3097, 3434, 3321, 1410, 2957, 2727, 1565, 3244, 2142, 1798, 1052, 1300, 1179, 1123, 1051, 676, 2024, 218, 2641, 843, 2314, 1, 789, 3919, 4402, 2023, 2066, 482, 1840, 5219, 638, 3262, 5092, 415, 2334, 4977, 4238, 4564, 1506, 72, 4962, 199, 363, 373, 2364, 4361, 799, 2240, 4180, 1629, 861, 4645, 2444, 3246, 4096, 2693, 3122, 5203, 631, 1748, 1334, 1765, 4935, 1604, 4828, 2058, 2317, 2633, 5138, 4190, 2681, 600, 4109, 3087, 1609, 3364, 2779, 3297, 4863, 1830, 1093, 4959, 832, 3045, 3404, 953, 2918, 892, 3346, 2646, 4215, 2450, 4557, 2917, 4771, 4630, 3182, 1613, 2176, 1275, 993, 3444, 1006, 3962, 4566, 3460, 1462, 537, 1842, 2500, 5168, 2051, 13, 3191, 1652, 2770, 3383, 2348, 2478, 3294, 4537, 5164, 4113, 1061, 110, 1720, 4313, 4218, 294, 1183, 4462, 1755, 3111, 586, 1249, 3674, 1188, 1715, 4926, 2127, 5195, 2543, 3625, 1619, 492, 2855, 1785, 3302, 4954, 3206, 3886, 4086, 4154, 382, 978, 1200, 248, 1942, 4211, 4602, 589, 3269, 3314, 4300, 3002, 2638, 4562, 2194, 2749, 1293, 703, 1796, 158, 1258, 2273, 4949, 635, 4623, 4887, 2957, 4756, 3280, 776, 4227, 94, 5246, 5105, 645, 3547, 3560, 4829, 354, 974, 3894, 4038, 3109, 3909, 1926, 2207, 2702, 1576, 1362, 2399, 4367, 2151, 5221, 2750, 3279, 4317, 959, 4868, 4487, 5162, 534, 2515, 3019, 4724, 3945, 3725, 1151, 1405, 790, 1531, 1426, 5150, 1607, 3670, 2782, 3140, 1908, 936, 1981, 2724, 1665, 4751, 934, 576, 3006, 4740, 1373, 3078, 4365, 1128, 1641, 2188, 1075, 1092, 1965, 4167, 2409, 295, 3106, 322, 47, 3432, 2440, 3319, 5146, 4582, 3960, 3728, 2292, 355, 2953, 2065, 2584, 1841, 4885, 2728, 2611, 2469, 4681, 1535, 5026, 4002, 143, 1113, 592, 4203, 3660, 4916, 3311, 1415, 3066, 4288, 2762, 1563, 3468, 4419, 2088, 328, 2481, 4174, 2933, 1836, 4869, 4843, 4519, 3442, 1464, 5269, 4020, 3522, 3164, 621, 1914, 2368, 4918, 385, 2095, 4838, 3757, 1304, 3884, 2809, 3478, 1502, 1346, 2035, 2797, 4450, 3993, 2713, 2619, 4235, 4534, 3796, 2027, 2070, 2231, 5233, 2752, 921, 3076, 1004, 1731, 1595, 1493, 2836, 3047, 4138, 1594, 3763, 1288, 52, 525, 3298, 5247, 16, 3053, 1440, 2439, 2180, 1181, 4573, 4721, 2015, 998, 770, 4076, 3849, 4781, 1811, 4890, 2295, 729, 1412, 1902, 5220, 5120, 5007, 4198, 2258, 4532, 3665, 523, 3093, 1494, 2835, 2664, 1333, 1368, 2438, 1475, 1148, 4135, 968, 2524, 2840, 2077, 5260, 244, 869, 477, 2935, 4767, 4700, 4522, 4728, 4985, 183, 4421, 2236, 1936, 214, 1217, 233, 69, 2304, 1108, 4643, 255, 4461, 2113, 1962, 2338, 4508, 1892, 2332, 132, 3937, 4208, 3215, 4086, 3320, 1094, 1187, 791, 5094, 1054, 3341, 3856, 188, 55, 1027, 3128, 1932, 1543, 3745, 78, 3038, 2279, 2561, 5038, 2879, 4238, 2263, 1141, 2281, 2205, 1564, 2944, 1759, 4669, 4381, 3643, 2958, 4563, 3746, 4934, 3749, 4981, 113, 3758, 3947, 4368, 395, 2248, 4558, 1832, 4741, 4608, 2950, 1616, 2346, 2711, 2475, 1032, 5080, 4031, 4216, 961, 1882, 3812, 2397, 4115, 489, 3677, 4370, 3312, 927, 2747, 3961, 1363, 3287, 4967, 2164, 2490, 4831, 2517, 2118, 4907, 4097, 4003, 4333, 1656, 3997, 731, 5208, 4371, 5060, 3334, 2196, 4008, 1752, 41, 3613, 1000, 1550, 2640, 1933, 488, 1611, 1517, 440, 955, 829, 4327, 2818, 4880, 682, 2492, 1813, 1236, 5016, 551, 4204, 3642, 2087, 1780, 361, 261, 273, 348, 4013, 3664, 310, 1408, 3971, 125, 1120, 4768, 4110, 669, 439, 2888, 3065, 4070, 4941, 1887, 3881, 4904, 312, 1468, 3479, 4788, 5010, 71, 4065, 2033, 2156, 714, 5223, 4555, 3144, 873, 4269, 453, 2642, 4406, 3145, 3166, 265, 2569, 1198, 1674, 4960, 163, 4094, 2884, 738, 3704, 4338, 1159, 231, 3783, 3694, 417, 4585, 2163, 667, 4506, 4657, 3514, 823, 5047, 1436, 2483, 4143, 2831, 5116, 5190, 1547, 1940, 2869, 3575, 5113, 4711, 4662, 1280, 646, 4217, 4753, 1376, 242, 48, 4334, 683, 3000, 4813, 5070, 2993, 1618, 3476, 4035, 3475, 2765, 2834, 3769, 5129, 2408, 5095, 2419, 81, 3307, 991, 4474, 1556, 518, 1050, 3650, 578, 5257, 1673, 3418, 2883, 230, 2700, 3848, 3341, 5119, 5216]
    }

CLASSES_MAP_EXTENDED = {
    "O", "not_medical_claim_nor_experience_nor_question",
    "claim", "medical_causal_claim",
    "per_exp", " medical_causal_experience",
    "claim_per_exp", "medical_causal_claim_based_on_experience",
    "question", "medical_question",
}

def load(
    mode=TaskTypeSemeval.TokenClassification, 
    data_option=SemevalDatasetSelection.Original, 
    classes_mapping=TargetClassesMapping.Original,
    test_split_size=TestSplitSize.Twenty, 
    verbose=False):
    """
    Loads full dataset from [Semeval 2023 Task 8.1](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN).
    """

    try:
        download("semeval_2023_t8.1")
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    path = datapath("semeval_2023_t8.1")
    
    X_train_raw = []
    X_train = []
    y_train_raw = []
    y_train=[]
    
    X_test_raw = []
    X_test = []
    y_test_raw = []
    y_test=[]
    
    if (data_option == SemevalDatasetSelection.Original):
        # load train
        with open(path / "st1_train.csv", "r") as fd:
            reader = csv.reader(fd)
            
            if (mode == TaskTypeSemeval.TokenClassification):
                X_train_raw, y_train_raw, X, y = load_tokens(reader, verbose)
                test_indices = set(TEST_SPLIT_SIZE_TO_INDEXES[test_split_size])
                
                X_test = [X[i] for i in test_indices]
                y_test = [y[i] for i in test_indices]
                
                X_train = [X[i] for i in range(len(X)) if i not in test_indices]
                y_train = [y[i] for i in range(len(y)) if i not in test_indices]
            
            if (mode == TaskTypeSemeval.SentenceClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_sentences(reader, True, verbose)
            
            if (mode == TaskTypeSemeval.SentenceMultilabelClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_sentences(reader, False, verbose)
            
        # load test
        # with open(path / "st1_test.csv", "r") as fd:
        #     reader = csv.reader(fd)
            
        #     if (mode == TaskTypeSemeval.TokenClassification):
        #         X_test_raw, y_test_raw, X_test, y_test = load_tokens(reader)
            
        #     if (mode == TaskTypeSemeval.SentenceClassification):
        #         X_test_raw, y_test_raw, X_test, y_test = load_sentences(reader)
            
        #     if (mode == TaskTypeSemeval.SentenceMultilabelClassification):
        #         X_test_raw, y_test_raw, X_test, y_test = load_sentences(reader, False)
                
    
    else:
        
        # load train
        with open(path / "st1_actual.csv", "r") as fd:
            reader = csv.reader(fd)
            
            if (mode == TaskTypeSemeval.TokenClassification):
                X_train_raw, y_train_raw, X, y = load_tokens(reader, verbose)
                test_indices = TEST_SPLIT_SIZE_TO_INDEXES[test_split_size]
                
                X_test = [X[i] for i in test_indices]
                y_test = [y[i] for i in test_indices]
                
                X_train = [X[i] for i in range(len(X)) if i not in test_indices]
                y_train = [y[i] for i in range(len(y)) if i not in test_indices]
            
            if (mode == TaskTypeSemeval.SentenceClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_sentences(reader, True, classes_mapping, verbose)
            
            if (mode == TaskTypeSemeval.SentenceMultilabelClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_sentences(reader, False, classes_mapping, verbose)
            
    return X_train, y_train, X_test, y_test
        
    
def load_tokens(reader, verbose=False):
    X_raw= []
    X = []
    y_raw = []
    y = []
    
    title_line = True
    invalid_rows = 0
    for row in reader:
        if title_line:
            title_line = False
            continue
        
        try:
            rawl = json.loads(row[2])
        except:
            if (verbose):
                print(f"Invalid row, annotation not recognized at line {reader.line_num}.")
            continue
            
        if (len(rawl) > 1 and verbose):
            print(f"Warning, multiple annotations detected in line {reader.line_num}!!")
            
        entities = rawl[0]['crowd-entity-annotation']['entities']
        
        text = row[3]
        
        if len(text) == 0: # Invalid row
            if (verbose):
                print(f"Invalid row, no text detected for line {reader.line_num}.")
            invalid_rows += 1
            continue
        
        if (text == '[deleted by user]\n[removed]'): # Invalid row
            if (verbose):
                print(f"Invalid row, referenced comment at line {reader.line_num} was deleted.")
            invalid_rows += 1
            continue
        
        tokens, iobtags = span_to_iob(text, entities)
        X_raw.append(text)
        y_raw.append(entities)
        
        assert len(tokens) == len(iobtags)
        
        X.append(tokens)
        y.append(iobtags)
        
        
    assert len(X) == len(y)
    assert len(X_raw) == len(y_raw)
    assert len(X_raw) == len(X)
    assert len(y_raw) == len(y)
    
    if (verbose):
        print(f"Loaded {len(X)} items. A total of {invalid_rows} rows were invalid.")
    
    for i in range(len(X)):
        assert len(X[i]) == len(y[i])
    
    return X_raw, y_raw, X, y
   
def load_sentences(reader, single_label = True, classes_mapping=TargetClassesMapping.Original, verbose=False):
    X_raw= []
    X = []
    y_raw = []
    y = []
    
    title_line = True
    invalid_rows = 0
    for row in reader:
        if title_line:
            title_line = False
            continue
        
        try:
            rawl = json.loads(row[2])
        except:
            if (verbose):
                print(f"Invalid row, annotation not recognized at line {reader.line_num}.")
            continue
            
        if (len(rawl) > 1 and verbose):
            print(f"Warning, multiple annotations detected in line {reader.line_num}!!")
            
        entities = rawl[0]['crowd-entity-annotation']['entities']
        
        text = row[3]
        
        if len(text) == 0: # Invalid row
            if (verbose):
                print(f"Invalid row, no text detected for line {reader.line_num}.")
            invalid_rows += 1
            continue
        
        if (text == '[deleted by user]\n[removed]'): # Invalid row
            if (verbose):
                print(f"Invalid row, referenced comment at line {reader.line_num} was deleted.")
            invalid_rows += 1
            continue
        
        sentences, labels = span_to_sentence_class(text, entities) if single_label else span_to_sentence_multilabel_class(text, entities)
        X.extend(sentences)
        y.extend(labels)
        
    assert X.count('') == 0
    assert len(X) == len(y)
    assert len(X_raw) == len(y_raw)
    
    if (verbose):
        print(f"Loaded {len(X)} items. A total of {invalid_rows} rows were invalid.")
    
    if (classes_mapping == TargetClassesMapping.Extended):
        y = [CLASSES_MAP_EXTENDED[iy] for iy in y]
    
    return X_raw, y_raw, X, y
        
def span_to_sentence_multilabel_class(text, entities):
    # Define the characters to split on to get sentences
    split_chars = [".", "!", "?", "\n"]
    
    # Split the text into sentences
    sentences = [sentence.strip() for sentence in re.split('|'.join(map(re.escape, split_chars)), text) if sentence.strip()]
    
    # Initialize the list of sentence labels
    sentence_labels = [[] for _ in sentences]
    
    # Assign labels to the sentences based on the entities they contain
    for entity in entities:
        start = entity['startOffset']
        end = entity['endOffset']
        label = entity['label']
        
        for i, sentence in enumerate(sentences):
            sentence_start = text.index(sentence)
            sentence_end = sentence_start + len(sentence)
            
            if (start >= sentence_start and start <= sentence_end) or (end >= sentence_start and end <= sentence_end):
                sentence_labels[i].append(label)
    
    # Convert the list of labels for each sentence to a set to remove duplicates
    sentence_labels = [list(set(labels)) for labels in sentence_labels]
    
    return sentences, sentence_labels

def span_to_sentence_class(text, entities):
    # Define the characters to split on to get sentences
    split_chars = [".", "!", "?", "\n"]
    
    # Sort the entities by start offset
    entities.sort(key=lambda x: x['startOffset'])
    
    # Initialize the list of sentences and sentence labels
    sentences = []
    sentence_labels = []
    
    # Initialize the current position in the text
    current_pos = 0
    
    for entity in entities:
        start = entity['startOffset']
        end = entity['endOffset']
        label = entity['label']
        
        # Add the text before the entity (if any) as separate sentences
        if start > current_pos:
            sentences_before = re.split('|'.join(map(re.escape, split_chars)), text[current_pos:start].strip().replace('\n', ' '))
            sentences.extend(sentence.strip() for sentence in sentences_before if sentence.strip())  # Filter out empty sentences
            sentence_labels.extend(['O' for sentence in sentences_before if sentence.strip()])
        
        # Extend the end of the entity to the next sentence boundary
        for sent_boundary in split_chars:
            end = text.find(sent_boundary, end)
            if (end != -1):
                break
            
        # If there's no sentence boundary, use the end of the text
        if end == -1:
            end = len(text)
        
        # Add the entity as a separate sentence
        entity = text[start:end].replace('\n', ' ').strip()
        if entity:
            sentences.append(text[start:end].replace('\n', ' ').strip())
            sentence_labels.append(label)
        
        # Update the current position
        current_pos = end
    
    # Add the text after the last entity (if any) as separate sentences
    if current_pos < len(text):
        sentences_after = re.split('|'.join(map(re.escape, split_chars)), text[current_pos:].strip().replace('\n', ' '))
        sentences.extend(sentence.strip() for sentence in sentences_after if sentence.strip())  # Filter out empty sentences
        sentence_labels.extend(['O' for sentence in sentences_after if sentence.strip()])
    
    return sentences, [ label if label is not None else 'None' for label in sentence_labels]

def span_to_iob(text, entities):
    # Define the characters to split on
    split_chars = ["\n", " ", ",", ".", ";", ":", "!", "?", "(", ")"]
    
    # Sort the entities by start offset
    entities.sort(key=lambda x: x['startOffset'])
    
    # Initialize the list of tokens and labels
    tokens = []
    labels = []
    
    # Initialize the current position in the text
    current_pos = 0
    
    for entity in entities:
        start = entity['startOffset']
        end = entity['endOffset']
        label = entity['label']
        
        # Tokenize the text before the entity (if any) and add it to the tokens and labels
        if start > current_pos:
            tokens_before = re.split('|'.join(map(re.escape, split_chars)), text[current_pos:start].strip())
            tokens_before = [token.strip() for token in tokens_before if token.strip()]
            tokens.extend(tokens_before)  # Filter out empty tokens
            labels.extend(['O'] * len(tokens_before))
        
        # Add the entity to the tokens and labels
        entity_tokens = text[start:end].split()
        tokens.extend(entity_tokens)
        labels.extend(([f'{label}'] if len(entity_tokens) > 0 else []) + [f'{label}'] * (len(entity_tokens) - 1))
        
        # Update the current position
        current_pos = end
    
    # Tokenize the text after the last entity (if any) and add it to the tokens and labels
    if current_pos < len(text):
        tokens_after = re.split('|'.join(map(re.escape, split_chars)), text[current_pos:].strip())
        tokens_after = [token.strip() for token in tokens_after if token.strip()]
        tokens.extend(tokens_after)  # Filter out empty tokens
        labels.extend(['O'] * len(tokens_after))
    
    return tokens, labels

def compare_tags(tag_list, other_tag_list):
    """
    compare two tags lists with the same tag format:

    (`tag_name`, `start_offset`, `end_offset`, `value`)
    """
    tags_amount = len(tag_list)

    if tags_amount != len(other_tag_list):
        print(
            "missmatch of amount of tags %d vs %d" % (tags_amount, len(other_tag_list))
        )
        return False

    tag_list.sort(key=lambda x: x[1])
    other_tag_list.sort(key=lambda x: x[1])
    for i in range(tags_amount):
        if len(tag_list[i]) != len(other_tag_list[i]):
            print("missmatch of tags format")
            return False

        for j in range(len(tag_list[i])):
            if tag_list[i][j] != other_tag_list[i][j]:
                print(
                    "missmatch of tags %s vs %s"
                    % (tag_list[i][j], other_tag_list[i][j])
                )
                return False

    return True


def get_qvals_plain(y, predicted):
    tp = 0
    fp = 0
    fn = 0
    total_sentences = 0
    for i in range(len(y)):
        tag = y[i]
        predicted_tag = predicted[i]
        if tag != "O":
            if tag == predicted_tag:
                tp += 1
            else:
                fn += 1
        elif tag != predicted_tag:
            fp += 1
            
        total_sentences += 1

    return tp, fp, fn, total_sentences

def leak_plain(y, predicted):
    """
    leak evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals_plain(y, predicted)
    try:
        return float(fn / total_sentences)
    except ZeroDivisionError:
        return 0.0

def precision_plain(y, predicted):
    """
    precision evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals_plain(y, predicted)
    try:
        return tp / float(tp + fp)
    except ZeroDivisionError:
        return 0.0

def recall_plain(y, predicted):
    """
    recall evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals_plain(y, predicted)
    try:
        return tp / float(tp + fn)
    except ZeroDivisionError:
        return 0.0

def F1_beta_plain(y, predicted, beta=1):
    """
    F1 evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    p = precision_plain(predicted, y)
    r = precision_plain(predicted, y)
    try:
        return (1 + beta**2) * ((p * r) / (p + r))
    except ZeroDivisionError:
        return 0.0

def basic_fn_plain(y, predicted):
    correct = 0
    total = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            total += 1

            _, tag = y[i][j]
            _, predicted_tag = predicted[i][j]
            correct += 1 if tag == predicted_tag else 0

    return correct / total

def macro_f1(y, predicted, *args, **kwargs):
    y_flat = [tag for sublist in y for tag in sublist]
    predicted_flat = [tag for sublist in predicted for tag in sublist]

    return macro_f1_plain(y_flat, predicted_flat)

def weighted_f1(y, predicted, *args, **kwargs):
    y_flat = [tag for sublist in y for tag in sublist]
    predicted_flat = [tag for sublist in predicted for tag in sublist]

    return weighted_f1_plain(y_flat, predicted_flat)

def macro_f1_plain(y, predicted, *args, **kwargs):
    """
    Macro-average F1 evaluation function
    """
    from sklearn.metrics import f1_score
    
    # Get the unique classes
    return f1_score(y, predicted, average='macro')

def weighted_f1_plain(y, predicted, *args, **kwargs):
    """
    Macro-average F1 evaluation function
    """
    from sklearn.metrics import f1_score
    
    # Get the unique classes
    return f1_score(y, predicted, average='weighted')

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
    TestSplitSize.Twenty: [3167, 3461, 770, 315, 2094, 3732, 2634, 3897, 1147, 865, 1181, 166, 3513, 882, 3706, 964, 3292, 69, 3253, 120, 3183, 3907, 1401, 3005, 1754, 736, 1567, 2179, 1219, 2626, 2927, 1253, 1649, 1329, 2542, 3022, 3825, 2019, 3593, 746, 1376, 687, 1192, 1027, 3536, 1386, 158, 3667, 3851, 2843, 770, 2964, 3513, 3556, 2836, 1380, 671, 3856, 168, 2820, 1448, 3725, 1406, 1703, 722, 2442, 416, 955, 964, 2436, 3095, 2302, 1067, 2982, 2153, 2018, 1002, 1369, 1959, 2790, 195, 1132, 3883, 3789, 875, 3693, 3660, 1811, 954, 946, 3383, 3147, 453, 3387, 980, 2203, 1576, 2051, 1827, 2871, 2055, 3371, 779, 1942, 1589, 162, 1301, 3603, 338, 1336, 2634, 2219, 3907, 3572, 1761, 871, 1242, 994, 1160, 185, 1446, 911, 3440, 1411, 1519, 3080, 2161, 1305, 2439, 3409, 3099, 2153, 1519, 2371, 363, 1730, 2985, 2024, 3531, 2199, 917, 2552, 3725, 3205, 558, 944, 3008, 1946, 1024, 206, 2999, 2541, 842, 3061, 2270, 3769, 1317, 2366, 137, 3448, 180, 527, 3823, 3371, 752, 3152, 223, 1630, 2924, 2759, 3830, 2970, 349, 1171, 1067, 2606, 3402, 1582, 717, 334, 3029, 2553, 3773, 386, 1710, 33, 1247, 176, 1750, 2014, 599, 3528, 1971, 1690, 281, 365, 203, 926, 275, 3444, 904, 2714, 1126, 2413, 198, 3029, 3349, 2711, 3124, 3284, 1448, 3028, 560, 3000, 2695, 765, 2310, 1352, 3900, 1363, 2011, 2625, 692, 2006, 2718, 3773, 386, 1920, 2978, 3548, 2325, 1416, 2006, 3337, 1619, 1389, 504, 2183, 997, 983, 3161, 2686, 2541, 1022, 1373, 378, 3034, 1355, 53, 2130, 2171, 1848, 1752, 344, 3523, 2288, 1507, 640, 1271, 2805, 164, 3076, 56, 3197, 1656, 2133, 2008, 2636, 2779, 2714, 2711, 598, 3744, 1580, 484, 889, 3572, 3480, 1359, 239, 3228, 572, 846, 3219, 1016, 2124, 3686, 1630, 3152, 3654, 1899, 1895, 3824, 543, 2709, 2552, 2778, 70, 740, 883, 1524, 3279, 330, 2833, 2222, 1348, 741, 3680, 2812, 2357, 1744, 1097, 959, 269, 1411, 893, 985, 1559, 585, 2495, 3194, 2324, 578, 3204, 549, 702, 1147, 2974, 3299, 3628, 1599, 2378, 3850, 1615, 15, 2683, 2792, 2674, 3885, 1190, 3460, 1840, 2263, 3195, 833, 3152, 546, 2727, 1697, 1982, 79, 897, 359, 2938, 984, 172, 3504, 3248, 663, 1913, 604, 1037, 576, 3889, 3430, 1304, 2652, 286, 478, 1392, 2334, 1977, 1554, 2029, 583, 1946, 739, 497, 1131, 1333, 899, 3367, 2349, 66, 1316, 582, 21, 3752, 131, 2969, 881, 694, 3073, 1205, 1177, 3123, 886, 186, 3785, 2549, 2161, 2985, 608, 520, 703, 226, 378, 129, 403, 2307, 1353, 1024, 497, 2325, 628, 1812, 2634, 3091, 56, 201, 2299, 3219, 2532, 1793, 747, 3780, 2035, 2013, 2960, 752, 2292, 3886, 11, 3419, 2400, 3861, 1607, 593, 2017, 932, 788, 749, 2655, 1153, 594, 3806, 1219, 1371, 2762, 2215, 626, 2267, 3873, 90, 1376, 2132, 2054, 2801, 2114, 2489, 1805, 773, 72, 3441, 17, 583, 594, 395, 439, 2557, 1687, 1651, 701, 2203, 3087, 738, 1569, 613, 2453, 481, 2034, 1287, 2096, 1753, 2716, 2680, 427, 26, 947, 1488, 3612, 3789, 3451, 1031, 1009, 2214, 1396, 1242, 3671, 1952, 606, 3203, 2837, 420, 585, 3300, 1373, 375, 1795, 810, 1441, 2318, 2711, 612, 2161, 1890, 1572, 2959, 2614, 559, 2650, 2460, 2538, 1437, 751, 1053, 1538, 937, 1255, 2443, 1172, 3040, 3540, 2279, 547, 1078, 1556, 2313, 902, 2118, 1076, 2612, 2362, 3536, 974, 3443, 1635, 157, 756, 1050, 825, 1489, 2901, 2496, 1069, 1101, 3506, 2913, 902, 3058, 1184, 2546, 3068, 1621, 3893, 3383, 767, 3048, 2614, 3266, 131, 3474, 303, 3037, 2511, 1727, 3883, 2842, 3754, 1602, 3293, 2532, 1671, 145, 360, 3893, 730, 3025, 2505, 4, 803, 1189, 3072, 2793, 2001, 3662, 2290, 1108, 954, 1758, 3842, 1947, 3277, 2892, 2220, 967, 2950, 1402, 3590, 739, 2181, 959, 840, 2498, 1815, 449, 2774, 1987, 3735, 188, 1513, 3359, 933, 850, 3600, 3358, 3596, 3049, 3743, 1787, 1028, 1414, 1191, 1557, 2397, 2022, 2609, 1827, 2434, 3356, 1576, 3680, 2795, 1992, 2594, 2079, 1481, 2470, 1269, 3785, 237, 1269, 3498, 758, 3211, 3470, 3850, 1445, 284, 1926, 447, 2479, 3345, 1664, 3004, 2137, 225, 2840, 1029, 2117, 1597, 3089, 2682, 1617, 3701, 3299, 3312, 2351, 3320, 121, 2422, 3458, 3016, 336, 1098, 2593, 1825, 3372, 2038, 3069, 542, 1517, 2359, 3444, 1618, 3261, 1553, 729, 1691, 3726, 1857, 3761, 2822, 1996, 3368, 3016, 818, 190, 1265, 1956, 2261, 769, 3272, 671, 85, 495, 3476, 3421, 1097, 686, 461, 1934, 715, 2726, 20, 1719, 984, 1192, 3171, 2702, 3858, 3133, 1234, 2778, 1444, 2558, 1846, 3247, 423, 2061, 1734, 1369, 1849, 3452, 3318, 3150, 640, 269, 891, 2253, 1268, 1094, 1875, 2795, 1535, 2978, 1822, 3173, 1764, 975, 3412, 1197, 3615, 1582, 1800, 2884, 587, 3668, 3308, 371, 2993, 3789, 74, 3761, 3047, 817, 3518, 227, 931, 1556, 3578, 1911],
}

CLASSES_MAP_EXTENDED = {
    "O", "not_medical_text",
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
                test_indices = TEST_SPLIT_SIZE_TO_INDEXES[test_split_size]
                
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
        pass

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

def macro_f1(y, predicted):
    y_flat = [tag for sublist in y for tag in sublist]
    predicted_flat = [tag for sublist in predicted for tag in sublist]

    return macro_f1_plain(y_flat, predicted_flat)

def macro_f1_plain(y, predicted):
    """
    Macro-average F1 evaluation function
    """
    # Get the unique classes
    classes = set(y)

    # Initialize the total precision and recall
    total_precision = 0
    total_recall = 0

    # Calculate precision and recall for each class
    for _class in classes:
        # Get the true positives, false positives, and false negatives for this class
        tp = sum([1 for tag, predicted_tag in zip(y, predicted) if tag == predicted_tag == _class])
        fp = sum([1 for tag, predicted_tag in zip(y, predicted) if tag != _class and predicted_tag == _class])
        fn = sum([1 for tag, predicted_tag in zip(y, predicted) if tag == _class and predicted_tag != _class])

        # Calculate precision and recall for this class
        try:
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
        except ZeroDivisionError:
            precision = recall = 0.0

        # Add the precision and recall to the totals
        total_precision += precision
        total_recall += recall

    # Calculate the macro-average precision and recall
    macro_precision = total_precision / len(classes)
    macro_recall = total_recall / len(classes)

    # Calculate the macro-average F1
    try:
        macro_f1 = 2 * ((macro_precision * macro_recall) / (macro_precision + macro_recall))
    except ZeroDivisionError:
        macro_f1 = 0.0

    return macro_f1

# load(mode=TaskTypeSemeval.TokenClassification, data_option=SemevalDatasetSelection.Actual)

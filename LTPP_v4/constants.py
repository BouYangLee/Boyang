# constants.py

# 谓词集合定义
mental_predicate_set = [1, 2, 3, 4]  # mental predicates
action_predicate_set = [5, 6, 7]        # action predicates
head_predicate_set = [1, 2, 3, 4, 5, 6, 7]  # all possible head predicates
total_predicate_set = mental_predicate_set + action_predicate_set  # [1, 2, 3, 4, 5, 6]

PAD = 0

# 时间网格长度
grid_length = 0.5

# 实验默认参数
DEFAULT_D = 6
DEFAULT_T_MATCH = 0.6  # 降低匹配温度，增强选择性
DEFAULT_TAU = 0.1      # 稍微增大tau，使softmin更平滑
DEFAULT_BETA = 5.0     # 降低beta，使soft-OR更包容

# 规则阈值参数
DEFAULT_G_THRESHOLD = 0.05
DEFAULT_DELTA_THRESHOLD = 1e-4
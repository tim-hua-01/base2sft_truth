"""
Configuration settings for deception detection probe training.
"""

# Task lists for different experiment configurations
TASK_LISTS = {
    'default': {
        'train': [
            # ### baseline 
            'got__best',
            'repe_honesty__plain',

            ### FLEED
            'claims__definitional_gemini_600_full', 'claims__evidential_gemini_600_full', 
            'claims__fictional_gemini_600_full', 'claims__logical_gemini_600_full', 
            'ethics__commonsense', 

            ### sycophancy 
            'sycophancy__mmlu_stem_same_conf_all',
            # 'sycophancy__mmlu_stem_conf_all',

            ### exp inverted
            'repe_honesty__IF_dishonest',

            ### combined
            'combined__claims_gemini600_ethical_sycosameconf_IFdishonest_RP_IT_SB', # MAIN
            
            ### honesty benchmark
            'roleplaying__plain',
            'insider_trading__upscale',
            'sandbagging_v2__wmdp_mmlu',

            # ### A&M 
            # 'internal_state__animals', 'internal_state__cities', 'internal_state__companies',
            # 'internal_state__elements', 'internal_state__facts', 'internal_state__inventions',
        ],
        'test': [
            ### FLEED
            'claims__definitional_gemini_600_full', 'claims__evidential_gemini_600_full',
            'claims__fictional_gemini_600_full', 'claims__logical_gemini_600_full',
            'ethics__commonsense',

            ### sycophancy
            'sycophancy__mmlu_stem_same_conf_all',
            # 'sycophancy__mmlu_stem_conf_all',

            ### exp inverted
            'repe_honesty__IF_dishonest',

            # ### Honesty benchmarks
            'roleplaying__plain',
            'insider_trading__upscale',
            'sandbagging_v2__wmdp_mmlu',
        ]
    },
}

# Combined dataset configurations
COMBINED_TASKS_CONFIG = {
    'combined__claims_gemini600_ethical_sycosameconf_IFdishonest_RP_IT_SB': { # final combined dataset
        'tasks': ['claims__definitional_gemini_600_full', 'claims__evidential_gemini_600_full',
                 'claims__logical_gemini_600_full', 'claims__fictional_gemini_600_full', 'ethics__commonsense',
                 'repe_honesty__IF_dishonest',
                 'roleplaying__plain', 'insider_trading__upscale', 'sandbagging_v2__wmdp_mmlu',
                 'sycophancy__mmlu_stem_same_conf_all',],
        'groups': [[0, 1, 2, 3, 4], [5], [6, 7, 8], [9]]
    },
}

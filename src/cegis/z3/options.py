from dataclasses import dataclass, field


@dataclass
class Options:
    """
      Options for configuring the Synthesis process.

    Attributes:
        logging_level (str): Logging level. Choices are DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.
        logging_file (str): Log file path. Default is logs/default.log.

    Synthesis Parameters:
        strategy (str): Synthesis strategy. Choices are random_enumerative, fast_enumerative. Default is random_enumerative.
        candidate_generation (str): Candidate generation strategy. Choices are bottom_up, top_down, fast_enumerative. Default is bottom_up.
        max_iterations (int): Maximum number of iterations. Default is 500.
        max_depth (int): Maximum depth of generated expressions. Default is 5.
        max_complexity (int): Maximum complexity of generated expressions. Default is 5.
        random_seed (int): Random seed for all solvers. Default is 1234.
        randomise_each_iteration (bool): Randomise seed between each synthesis iteration. Default is False.
        max_candidates_at_each_depth (int): Maximum number of candidate programs to consider at each depth. Default is 10.
        min_const (int): Minimum constant to introduce into the candidate programs. Default is -2.
        max_const (int): Maximum constant to introduce into the candidate programs. Default is 2.
        operation_costs (Dict[str, int]): Key-Value pairs representing operation costs for adjustable weighting. 
                                          Default is {'+': 1, '-': 1, '*': 2, 'If': 3, 'Neg': 1}.
        initial_weight (float): Initial weight for operations in random candidate generation. Default is 1.0.
        weight_multiplier (float): Weight multiplier for operations in candidate generation. 
                                   1 = no diversity, >1 will bias towards previously chosen operations and <1 will bias away. Default is 1.0.

    Solver Parameters:
        name (str): SMT Solver to use. Choices are z3, cvc5 (experimental/alpha phase). Default is z3.
        timeout (int): SMT Solver Configuration - Timeout. Default is 30000.

    Other:
        input_source (str): Input source. Choices are STDIN or a path to a problem file.

    Note:
        When using a YAML configuration file, these options can be grouped under 'logging', 'synthesis_parameters', and 'solver' sections.
        For example:

        logging:
          level: "DEBUG"
        synthesis_parameters:
          max_iterations: 20
          operation_costs:
            '+': 1
            '-': 1
            '*': 3
            'If': 4
            'Neg': 2
        solver:
          name: "z3"
          timeout: 30000
    """

    logging_level: str = field(
        default="INFO",
        metadata=dict(
            description="Logging level",
            type="str",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ))
    logging_file: str = field(
        default="logs/default.log",
        metadata=dict(
            description="Log file path",
            type="str"
        ))
    synthesis_parameters_strategy: str = field(
        default="random_enumerative",
        metadata=dict(
            description="Synthesis strategy",
            type="str",
            choices=["random_enumerative", "fast_enumerative"]
        ))
    synthesis_parameters_candidate_generation: str = field(
        default="bottom_up",
        metadata=dict(
            description="Candidate generation strategy",
            type="str",
            choices=["bottom_up", "top_down", "fast_enumerative"]
        ))
    synthesis_parameters_max_iterations: int = field(
        default=500,
        metadata=dict(
            description="Maximum number of iterations",
            type="int"))
    synthesis_parameters_max_depth: int = field(
        default=5,
        metadata=dict(
            description="Maximum depth of generated expressions",
            type="int"
        ))
    synthesis_parameters_max_complexity: int = field(
        default=5,
        metadata=dict(
            description="Maximum complexity of generated expressions",
            type="int"
        ))
    synthesis_parameters_random_seed: int = field(
        default=1234,
        metadata=dict(
            description="Random seed for all solvers",
            type="int"
        ))
    synthesis_parameters_randomise_each_iteration: bool = field(
        default=False,
        metadata=dict(
            description="Randomise seed between each synthesis iteration",
            type="bool"
        ))
    synthesis_parameters_max_candidates_at_each_depth: int = field(
        default=10,
        metadata=dict(
            description="Maximum number of candidate programs to consider at each depth",
            type="int"
        ))
    synthesis_parameters_min_const: int = field(
        default=-2,
        metadata=dict(
            description="Minimum constant to introduce into the candidate programs",
            type="int"
        ))
    synthesis_parameters_max_const: int = field(
        default=2,
        metadata=dict(
            description="Maximum constant to introduce into the candidate programs",
            type="int"
        ))
    synthesis_parameters_operation_costs: dict[str, int] = field(
        default_factory=lambda: {'+': 1, '-': 1, '*': 2, 'If': 3, 'Neg': 1},
        metadata=dict(
            description="Operation costs for candidate generation",
            type="dict"
        )
    )
    synthesis_parameters_initial_weight: float = field(
        default=1.0,
        metadata=dict(
            description="Initial weight for operations in random candidate generation",
            type="float"
        ))
    synthesis_parameters_weight_multiplier: float = field(
        default=1.0,
        metadata=dict(
            description="Weight multiplier for operations in candidate generation. 1 = no diversity, >1 will bias towards previously chosen operations and <1 will bias away",
            type="float"
        ))

    solver_name: str = field(
        default="z3",
        metadata=dict(
            description="SMT Solver to use",
            type="str",
            choices=["z3", "cvc5"]
        ))
    solver_timeout: int = field(
        default=30000,
        metadata=dict(
            description="SMT Solver Configuration - Timeout",
            type="int"
        ))

    input_source: str = field(
        default="stdin",
        metadata=dict(
            description="Source of the input problem (stdin or file path)",
            type="str"
        ))

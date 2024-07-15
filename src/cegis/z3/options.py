from dataclasses import dataclass, field


@dataclass
class Options:
    """
      Options for configuring the Synthesis process.

      Attributes:
        logging_level (str): Logging level. Choices are DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.
        logging_file (str): Log file path. Default is logs/default.log.
        synthesis_parameters_strategy (str): Synthesis strategy. Choices are random_enumerative, fast_enumerative. Default is random_enumerative.
        synthesis_parameters_max_iterations (int): Maximum number of iterations. Default is 500.
        synthesis_parameters_max_depth (int): Maximum depth of generated expressions. Default is 5.
        synthesis_parameters_max_complexity (int): Maximum complexity of generated expressions. Default is 5.
        synthesis_parameters_random_seed (int): Random seed for all solvers. Default is 1234.
        synthesis_parameters_randomise_each_iteration (bool): Randomise seed between each synthesis iteration. Default is False.
        synthesis_parameters_max_candidates_at_each_depth (int): Maximum number of candidate programs to consider at each depth. Default is 10.
        synthesis_parameters_min_const (int): Minimum constant to introduce into the candidate programs. Default is -2.
        synthesis_parameters_max_const (int): Maximum constant to introduce into the candidate programs. Default is 2.
        solver_name (str): SMT Solver to use. Choices are z3, cvc5. Default is z3.
        solver_timeout (int): SMT Solver Configuration - Timeout. Default is 30000.
        input_source (str): Input source. Choices are STDIN or a path to a problem file.
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

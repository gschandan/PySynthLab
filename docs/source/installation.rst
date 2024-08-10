Installation
============

Prerequisites
-------------

Before installing PySynthLab, ensure you have at least the following installed:

- Python 3.8 or higher
- Virtualenv recommended
- Z3 Theorem Prover

Installation Steps
------------------

1. Clone the repository:

   .. code-block:: shell

      git clone https://github.com/gschandan/PySynthLab.git
      cd PySynthLab

2. Create and activate the virtual environment:

   .. code-block:: shell

      python -m venv venv
      source venv/bin/activate

3. Install the requirements:

   You have several options to install the requirements:

   a. Using pip without venv:

      .. code-block:: shell

         pip install -r requirements.txt

   b. Using Make:

      .. code-block:: shell

         make init

   c. Using pip with venv:

      .. code-block:: shell

         ./venv/bin/pip install -r requirements.txt

   d. Using Make with venv:

      .. code-block:: shell

         make init_venv

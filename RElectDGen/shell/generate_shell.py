
def generate_shell(config):
    
    if config.get('machine','arjuna')=='arjuna':
        from .generate_shell_arjuna import shell_from_config
    elif config.get('machine')=='bridges':
        from .generate_shell_bridges import shell_from_config

    shell_from_config(config)
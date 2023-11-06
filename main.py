from config import parser
from utils import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config
from trainers import base_procedure


def main():
    # initial
    args = parser.parse_args()
    args = vars(args)
    args = load_config(args)
    
    if args['log'] is True:
        saving_path, saving_name = set_up_logger(args["model"], args["dataset"], args['log_dir'])
    elif args.save_best is True or args.save_check_points is True:
        saving_path, saving_name = get_dir_path(args["model"], args["dataset"], args['log_dir'])
    else:
        saving_path, saving_name = None, None
    
    args['saving_path'] = saving_path
    args['saving_name'] = saving_name
    save_config(args, saving_path)
    set_device(args['cuda'], args['device'])
    set_seed(args['random_seed'])
    
    base_procedure(args)
    

if __name__ == "__main__":
    main()

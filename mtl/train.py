"""train.py
Trains models for multiple multimodal tasks using dataset inside mtl/data.

Usage examples (training model):
    python train.py \
        --exp_name test_bc \
        --task imitation \
        --n_epochs 10 \
        --commands Fly_to_the_tree_nearby_gray_vehicle
    python train.py \
        --exp_name test_joint_training \
        --task joint_training \
        --n_epochs 10 \
        --commands Fly_to_the_tree_nearby_gray_vehicle

Usage example (ablation study, train on all tasks)
    python3 train.py \
        --exp_name baseline_bc
        --task joint_training
        --n_epochs=10
        --weights 0 1

Usage examples (visualizing gaze predictions, model already trained):
    python train.py \
        --exp_name test_joint_training \
        --viz_gaze \
        --model_trained

Usage example (training agent at Truck Mountains env)
    python train.py \
        --exp_name truck_mountains_v6 \
        --command Fly_to_the_nearest_truck \
        --n_epochs 50 \
        --split_ratio 0.7 \
        --weights 1 2

    python train.py \
        --exp_name interv_model \
        --command Fly_to_the_nearest_truck \
        --n_epochs 50 \
        --split_ratio 0.7 \
        --weights 1 3
"""
from imports import *
from utils import *
from samplers import *
from losses import *
from models import *
from preprocess_data import *
from common_utils import *

import models

gym_path = os.getcwd().split('mtl')[0]+'airsim/'
sys.path.insert(0, gym_path)

from gym_log_airsim import run_rollouts


#################### Configuration ##########

my_parser = argparse.ArgumentParser(
    prog='train.py', usage='%(prog)s [options] path',
    description='Multimodal learning.')
my_parser.add_argument('--task', type=str, default='joint_training', \
    help='choose joint_training for training policy model or depth for training DepthVAE')
my_parser.add_argument('-c', '--commands', type=str, nargs='+', default=[],  \
    help='Enter whitespace separated commands, whitespace within commands should be substituted with underscores, leave empty to use all')
my_parser.add_argument('-w', '--weights', type=int, nargs='+', default=[1, 1], \
    help='Enter loss weights')
my_parser.add_argument('--lr', type=float, default=3e-4)
my_parser.add_argument('--split_ratio', type=float, default=0.8)
my_parser.add_argument('--n_epochs', type=int, default=10)
my_parser.add_argument('--batch_size', type=int, default=128)
my_parser.add_argument('--n_hidden', type=int, default=32)
my_parser.add_argument('--rnd_seed', type=int, default=0)
my_parser.add_argument('--model_trained', action='store_true')
my_parser.add_argument('--run_tests', action='store_true')
my_parser.add_argument('--resume_experiment', action='store_true')
my_parser.add_argument('--gaze_condition', type=str, default='gazebc')
my_parser.add_argument('--training_fraction', type=float, default=0.8)
args = my_parser.parse_args()

# randomize seeds
random.seed(args.rnd_seed)
np.random.seed(args.rnd_seed)
torch.manual_seed(args.rnd_seed)


def get_data_objects(task, df=None, train_ids=None, test_ids=None, commands=[]):
    if task != 'depth':
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # tensors are too small for any GPU benefit, CPU-GPU transfers take more time
        logger.info(f'Using {device} device')

        df_train, df_test = get_dataframes(df, train_ids, test_ids)

        # sampler = None

        # for creating batches with samples from as many classes as possible
        # sampler = StratifiedSampler(class_vector=df_train['command'], batch_size=bs)

        # for handling imbalanced action data
        sampler = get_BC_sampler(df_train)

        # create training and validation datasets, episodes are split between train and val instead of individual points
        train_dataset = AirsimDataset(data=df_train)
        test_dataset = AirsimDataset(data=df_test)
    else:
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Using {device} device')

        sampler = None
        init_dataset = DepthDataset(path='data/')
        split_ratio = 0.8
        lengths = [int(len(init_dataset)*split_ratio), len(init_dataset)-int(len(init_dataset)*split_ratio)]
        train_dataset, test_dataset = random_split(init_dataset, lengths)

    return train_dataset, test_dataset, sampler, device

def get_training_objects(task, sample_batch, device, weights=None, n_hidden=None):

    if task == 'grounding':

        ########### For language grounding #################################

        '''
        Use either the triplet selector block OR the pair selector block

        1) Creates triplets, usually a very high number of them (not recommended)
        OR
        2) Creates balanced number of positive and negative pairs. Either use this or the triplet method above.
        '''

        triplet_selector = AllTripletSelector()
        loss_fn = OnlineTripletLoss(margin=10, triplet_selector=triplet_selector)

        # pair_selector = AllPositivePairSelector()
        # loss_fn = OnlineContrastiveLoss(margin=10, pair_selector=pair_selector)

        input_dim = sample_batch['obs'].size(-1)

        model_type = 'MEM'
        model_params = {'input_dim':input_dim, 'use_batch_norm':True}
        metrics = {'loss':Loss(loss_fn)}

    elif task == 'gazepred':
        # predicting only means and assuming unit variance
        loss_fn = GazeKLDUnit()

        input_dim = sample_batch['obs'].size(-1)

        model_type = 'GazeModelUnit'
        model_params = {'input_dim':input_dim, 'n_hidden':n_hidden}
        metrics = {'loss':Loss(loss_fn)}

    elif task == 'joint_training':

        loss_fn = JointLoss(weights=weights)
        gaze_loss_fn = JointLoss(weights=[1,0])
        bc_loss_fn = JointLoss(weights=[0,1])

        action_dim = sample_batch['actions'].size(-1)

        model_type = 'JointModel'
        model_params = {'action_dim':action_dim, 'n_hidden':n_hidden, 'num_commands':3}
        metrics = {'loss':Loss(loss_fn), 'loss_gaze':Loss(gaze_loss_fn), 'loss_bc':Loss(bc_loss_fn)}

    elif task == 'imitation':

        loss_fn = nn.MSELoss()

        input_dim = sample_batch['obs'].size(-1)
        action_dim = sample_batch['actions'].size(-1)

        model_type = 'AirSimBCModel'
        model_params = {'input_dim':input_dim, 'output_dim':action_dim}
        metrics = {'loss':Loss(loss_fn)}

    elif task == 'depth':
        loss_fn = VAE_loss()

        model_type = 'VAE'
        model_params = {'image_channels':1}
        metrics = {'loss':Loss(loss_fn)}

    else:
        raise NotImplementedError

    model = getattr(models, model_type)(**model_params)
    # trying different weight initialization schemes
    model.apply(model.weights_init)
    model = model.to(device)

    # if torch.cuda.device_count() > 1:
    #     model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    config = dict()
    config.update({
        'model_type': model.__class__.__name__,
        'model_params': model_params
        })

    return model, config, loss_fn, metrics

def train(**params):
    n_epochs = params['n_epochs']
    lr = params['lr']
    bs = params['batch_size']
    weights = params['weights']
    task = params['task']
    writer = params['writer'] 
    logdir = params['experiment_folder']

    # defaultdict avoids missing key error if depth training is done, as then train/test ids are not provided
    params = defaultdict(int, params)
    
    train_dataset, test_dataset, sampler, device = get_data_objects(
        task=task, 
        df=params['df'], 
        train_ids=params['train_epis'][params['training_fraction']], 
        test_ids=params['test_epis'], 
        commands=params['commands'])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=bs,
        shuffle=True if not sampler else False, 
        sampler=sampler,
        drop_last=True)

    val_loader = DataLoader(
        test_dataset, 
        batch_size=bs,
        shuffle=False,
        drop_last=True)

    logger.info(f'Number of training samples {len(train_dataset)}')
    logger.info(f'Number of testing samples {len(test_dataset)}')
    logger.info(f'Weights: {weights}')

    # use a sample batch to capture info about the dataset (n_inputs, n_outputs, etc)
    sample_batch = next(iter(train_loader))

    model, config, loss_fn, metrics = get_training_objects(
        task=task, 
        sample_batch=sample_batch, 
        device=device, 
        weights=weights)
    
    # dump model config to be later used by AirSim BC agent    
    with open(f'{logdir}/config.txt', 'w') as f:
        f.write(str(config))

    optimizer = Adam(model.parameters(), lr=lr)

    def log_scalar(name, value, step):
        """Log a scalar value to both MLflow and TensorBoard"""
        writer.add_scalar(name, value, step)
        mlflow.log_metric(name, value, step)

    def garbage_collect():
        for obj in gc.get_objects():   # Browse through ALL objects
            try:
                if isinstance(obj, h5py.File):   # Just HDF5 files
                    try:
                        obj.close()
                    except:
                        pass # Was already closed
            except:
                pass # some weird object

    # helper function to plot BC errors and show associated RGB frame
    def predict_on_batch(engine, batch):
        nonlocal model
        model.eval()
        model = model.to('cpu')
        output = {'loss_bc':list(), 'files':list()}
        pred_actions = model(batch)[-1]
        
        for i in range(batch['visual_features'].size(0)):
            # pytorch dataloader performs automatic collation
            true_action = batch['actions'][i]
            output['loss_bc'].append(((pred_actions[i]-true_action)**2).mean().item())
            output['files'].append(batch['camera_frame_file'][i])

        return output

    def plot_gaze(engine, batch):
        nonlocal model
        model.eval()
        model = model.to('cpu')
        gaze_outputs = model(batch)[0]

        # write batched gaze_outputs to csv file
        data_dict = {
            'iteration': engine.state.iteration,
            'frame_addr': batch['camera_frame_file'],
            'gaze_outputs_x': gaze_outputs.detach().numpy()[:,0],
            'gaze_outputs_y': gaze_outputs.detach().numpy()[:,1],
        }
        df_log = pd.DataFrame(data_dict)
        os.makedirs(f'{logdir}/gaze_on_demo/', exist_ok=True)
        df_log.to_csv(f'{logdir}/gaze_on_demo/test_gaze_iter{engine.state.iteration}.csv')


    def preprocess_batch(batch):
        # depends on task type
        if task == 'gazepred':
            return batch['obs'], batch['gaze_coords']
        elif task == 'grounding':
            return batch['obs'], batch['command']
        elif task == 'imitation':
            return batch['obs'], batch['actions']
        elif task == 'joint_training':
            return batch, (batch['gaze_coords'], batch['actions'])
        elif task == 'depth':
            return batch, batch
        else:
            raise NotImplementedError

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    terminal_evaluator = Engine(predict_on_batch)

    gaze_plotter = Engine(plot_gaze)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=False)
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.STARTED)
    def plot_graph(engine):
        # writer.add_graph(model, engine.state.batch)
        pass

    @trainer.on(Events.ITERATION_STARTED)
    def switch_batch(engine):
        engine.state.batch = preprocess_batch(engine.state.batch)

    @evaluator.on(Events.ITERATION_STARTED)
    def switch_batch(engine):
        engine.state.batch = preprocess_batch(engine.state.batch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        
        for key in metrics.keys():
            log_scalar(f'train/{key}', metrics[key] , trainer.state.epoch)
            logger.info(f'Training Results - Epoch[{trainer.state.epoch}] {key}: {metrics[key]:.2f}')
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics        

        for key in metrics.keys():
            log_scalar(f'val/{key}', metrics[key] , trainer.state.epoch)
            logger.info(f'Validation Results - Epoch[{trainer.state.epoch}] {key}: {metrics[key]:.2f}')

        trainer.state.score = -metrics['loss']

        pbar.n = pbar.last_print_n = 0

    @terminal_evaluator.on(Events.ITERATION_COMPLETED)
    def plot_errors(engine):
        trainer.pred_errors['errors'].extend(engine.state.output['loss_bc'])
        trainer.pred_errors['files'].extend(engine.state.output['files'])

    @trainer.on(Events.COMPLETED)
    def plot_pred_errors(trainer):
        if task is not 'joint_training':
            return

        load_best_model(model, f'../mtl/{logdir}/')

        gaze_plotter.run(val_loader)

        trainer.pred_errors = {'errors':list(), 'files':list()}
        terminal_evaluator.run(val_loader)

        save_obj(trainer.pred_errors, 'pred_errors')


    def score_function(engine):
        return engine.state.score

    es_handler = EarlyStopping(
        patience=50,
        score_function=score_function,
        trainer=trainer)
    save_handler = ModelCheckpoint(
        dirname=logdir,
        filename_prefix=f'{exp_cat}',
        require_empty=False,
        score_function=score_function,
        score_name='neg_val_loss',
        n_saved=10,
        atomic=True,
        save_as_state_dict=True,
        create_dir=True)

    # evaluator.add_event_handler(Events.COMPLETED, es_handler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'model':model})

    trainer.run(train_loader, max_epochs=n_epochs)

    garbage_collect()

exp_cat = f'GazeBC vs Vanilla BC'
try:
    experiment_id = mlflow.create_experiment(name=exp_cat)
except:
    experiment_id = mlflow.get_experiment_by_name(exp_cat).experiment_id  # if Experiment ID already exists

with mlflow.start_run(experiment_id=experiment_id, run_name=f'{exp_cat}_{datetime.now()}'):

    # fractions of full dataset to be used for training
    training_fractions = [0.9, 0.8, 0.6, 0.4, 0.2, 0.1]
    training_fraction = [args.training_fraction]

    if args.gaze_condition == 'vanillabc':
        weight_condition = [[0, 1]]
    elif args.gaze_condition == 'gazebc':
        weight_condition = [[1, 1]]

    if args.resume_experiment:
        df = load_obj('df')
        train_epi_ids_dict = load_obj('train_epi_ids_dict')
        test_epi_ids = load_obj('test_epi_ids')
    else:
        # create the master index table
        df = create_df(folder='data/', commands=args.commands)

        # create dictionary of episode lists used for runs associated with the training fractions
        # NOTE: below fn ensures that episodes for smaller fractions are subsets of episode list for larger fractions
        train_epi_ids_dict, test_epi_ids = select_episodes(df, training_fractions)

        save_obj(df, 'df')
        save_obj(train_epi_ids_dict, 'train_epi_ids_dict')
        save_obj(test_epi_ids, 'test_epi_ids')

    vars(args).update({'df':df,'train_epis':train_epi_ids_dict,'test_epis':test_epi_ids})

    logger = get_logger('TRAINING_LOGGER')

    if not args.model_trained:

        # can nest another loop of runs over here with changing epoch numbers etc.

        ## EXPERIMENTS
        # 1) [STATIC] evaluate performance of gazeBC vs vanillaBC model based on number of demonstrations
        param_configs = product_dict(**{
            'n_epochs':[args.n_epochs],
            'lr':[args.lr],
            'weights': weight_condition, #[1, 2]
            'training_fraction': training_fraction,
            'dynamic_target': [False]})

        # # 2) [DYNAMIC] evaluate performance of gazeBC vs vanillaBC model based on number of demonstrations
        # param_configs = product_dict(**{
        #     'n_epochs':[args.n_epochs],
        #     'lr':[args.lr],
        #     'weights': [[1, 2], [0, 1]],
        #     'training_fraction': training_fractions,
        #     'dynamic_target': [True]})

        # single experiment default run
        # param_configs = {'use_argparse':True}

        for item in param_configs:
            
            with mlflow.start_run(experiment_id=experiment_id, run_name=f'{exp_cat}_{item}', nested=True):
                
                logger.info(f'Running training with config {item}')
                # print('Weights:', item['weights'])

                logdir = f'logs/{datetime.now()}'
                writer = SummaryWriter(log_dir=logdir)

                vars(args).update(item)
                vars(args).update({'experiment_folder':logdir, 'writer':writer})
                
                train(**vars(args))

                logger.info('Saving experiment run configuration in MLflow')
                for key, value in vars(args).items():
                    mlflow.log_param(key, value)

                logger.info(f'>> Finished run')

                writer.close()

                logger.info('Uploading TensorBoard events as a run artifact.')
                mlflow.log_artifacts(logdir, artifact_path='events')

                if not args.run_tests:
                    continue

                num_attempts = 0
                evaluation_done = False

                # try at max 3 times, to run the airsim rollout test
                # while not evaluation_done and num_attempts < 3:  
                #     try:
                test_metrics = run_rollouts(
                    env_name='MultimodalAirSimMountains-v0',
                    exp_name=logdir,
                    n_steps=1000,
                    command='Fly to the nearest truck',
                    use_joystick=False,
                    df=df,
                    test_epi_ids=test_epi_ids,
                    start_epi=0,
                    max_episodes=100,
                    dynamic_target=item['dynamic_target'])

                for key, value in test_metrics.items():
                    mlflow.log_metric(key, value)

                evaluation_done = True
                logger.info(f'Completed test for model with config {item} for {num_attempts} time')

                    # except Exception as e:
                    #     print('ERROR:', e)
                    #     num_attempts += 1
                    #     logger.info(f'Could not run test for model with config {item} for {num_attempts} time')


    logger.info(f'>> Finished all runs for {exp_cat}')





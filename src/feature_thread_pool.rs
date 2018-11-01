use std;
use std::collections::HashSet;
use std::mem::replace;
use std::sync::Arc;

use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::sync::mpsc::SyncSender;
use std::sync::mpsc::sync_channel;

use std::thread;

extern crate rand;

use smallvec::SmallVec;

use rank_vector::RankVector;
use rank_vector::Node;
use io::DispersionMode;

impl FeatureThreadPool{
    pub fn new(size: usize) -> Sender<FeatureMessage> {

        if size < 1 {
            panic!("Warning, no processors were allocated to the pool, quitting!");
        }

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(size);

        for i in 0..size {

            workers.push(Worker::new(i,worker_receiver_channel.clone()))

        }

        tx
    }

    pub fn terminate(channel: &mut Sender<FeatureMessage>) {
        while let Ok(()) = channel.send(FeatureMessage::Terminate) {};
    }

}


pub struct FeatureThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<FeatureMessage>>>,
    sender: Sender<FeatureMessage>
}


impl Worker{

    pub fn new(id:usize,channel:Arc<Mutex<Receiver<FeatureMessage>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {

                let mut local_vector = RankVector::<SmallVec<[Node;1024]>>::empty_sv();

                loop{
                    let message_option = channel.lock().unwrap().recv().ok();
                    if let Some(message) = message_option {
                        match message {
                            FeatureMessage::Message((vector,draw_order,drop_set,dispersion_mode),sender) => {
                                sender.send(compute(vector,draw_order,drop_set,dispersion_mode,&mut local_vector)).expect("Failed to send feature result");
                            },
                            FeatureMessage::Terminate => break
                        }
                    }
                }
            }),
        }
    }
}

struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
    // worker_receiver_channel: Arc<Mutex<Receiver<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>>>,
}


pub enum FeatureMessage {
    Message((Arc<RankVector<Vec<Node>>>,Arc<Vec<usize>>,Arc<HashSet<usize>>,DispersionMode), mpsc::Sender<Vec<f64>>),
    Terminate
}

fn compute (prot_vector: Arc<RankVector<Vec<Node>>> , draw_order: Arc<Vec<usize>> , drop_set: Arc<HashSet<usize>>, dispersion_mode:DispersionMode, local_vector: &mut RankVector<SmallVec<[Node;1024]>>) -> Vec<f64> {

    local_vector.clone_from_prototype(&prot_vector);

    let result = match dispersion_mode {
        DispersionMode::Variance => local_vector.ordered_variance(&draw_order,&drop_set),
        DispersionMode::MAD => local_vector.ordered_mads(&draw_order,&drop_set),
        DispersionMode::SSME => local_vector.ordered_ssme(&draw_order, &drop_set),
        DispersionMode::Mixed => panic!("Mixed mode not a valid split setting for individual trees!"),
    };

    // println!("parallel: {:?}", result);

    result

}

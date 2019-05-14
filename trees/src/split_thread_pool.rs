use std;
use std::collections::HashSet;
use std::sync::Arc;

use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use std::thread;

extern crate rand;


use crate::rank_table::RankTable;
use crate::feature_thread_pool::FeatureThreadPool;
use crate::feature_thread_pool::FeatureMessage;


impl SplitThreadPool{
    pub fn new(processors: usize) -> Sender<SplitMessage> {

        if processors < 1 {
            panic!("Warning, no processors were allocated to the pool, quitting!");
        }

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(processors);

        for i in 0..((processors/4).max(1)) {

            workers.push(Worker::new(i,4.min(processors),worker_receiver_channel.clone()))

        }

        tx
    }

    pub fn terminate(channel: &mut Sender<SplitMessage>) {
        while let Ok(()) = channel.send(SplitMessage::Terminate) {};
    }

}


pub struct SplitThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<SplitMessage>>>,
    sender: Sender<SplitMessage>
}


impl Worker{

    pub fn new(id:usize, pool_limit:usize, channel:Arc<Mutex<Receiver<SplitMessage>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {
                let mut pool = FeatureThreadPool::new(5.min(pool_limit));
                loop{
                    let message_option = channel.lock().unwrap().recv().ok();
                    if let Some(message) = message_option {
                        match message {
                            SplitMessage::Message((prot_table,draw_order,drop_set,weights),sender) => {
                                sender.send(compute(prot_table,draw_order,drop_set,weights,pool.clone())).expect("Failed to send feature result");
                            },
                            SplitMessage::Terminate => {
                                FeatureThreadPool::terminate(&mut pool);
                                break
                            }
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


pub enum SplitMessage {
    Message((Arc<RankTable>,Vec<usize>,HashSet<usize>,Vec<f64>), mpsc::Sender<Option<(usize,usize,f64)>>),
    Terminate,
}

fn compute (prot_table: Arc<RankTable>, draw_order: Vec<usize> , drop_set: HashSet<usize>,weights: Vec<f64>, pool: Sender<FeatureMessage>) -> Option<(usize,usize,f64)> {

    // println!("Computing in split thread pool");

    prot_table.parallel_split_order_min(&draw_order,&drop_set,Some(&weights),pool)

    // println!("parallel: {:?}", result);
}

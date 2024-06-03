use std::cmp::max;
use std::ops::{Add, Sub};
use std::thread::sleep;
use std::time::Duration;
use chrono::{Local, Duration as ChronoDuration, DateTime};
use rand::Rng;
use rusqlite::{Connection, Result, RowIndex};
use serde::{Deserialize, Serialize};
use crate::advanced_problem::AdvancedHomeProblem;
use crate::basic_problem::{ApplianceAction, BatteryAction, HomeProblem};
use crate::planner::astar;
use crate::thread::ThreadPool;

mod data;
mod planner;
mod basic_problem;
mod extended_problem;
mod advanced_problem;
mod thread;

#[derive(Debug)]
struct Problem {
    id: u32,
    data: String,
}

#[derive(Serialize, Deserialize)]
struct Action {
    battery: i32,
    appliances: Vec<u32>,
}

fn random_solution(horizon: u32, num_appliances: u32) -> Vec<Action> {
    let mut rng = rand::thread_rng();
    let mut solution = vec![];
    for _ in 0..horizon {
        let mut appliance_actions = vec![];
        for _ in 0..num_appliances {
            appliance_actions.push(rng.gen_range(0..2));
        }
        solution.push(Action {
            battery: rng.gen_range(-1..2),
            appliances: appliance_actions,
        });
    }
    return solution;
}

fn retrieve_problems(db_path: String, started_at: DateTime<Local>) -> Result<Vec<Problem>> {
    let mut problems = vec![];

    let connection = Connection::open(db_path)?;

    let retrieved_at = Local::now();
    let timeout = max(started_at, retrieved_at - ChronoDuration::minutes(15));
    let data = (&retrieved_at.to_rfc3339(), &timeout.to_rfc3339());
    let mut statement = connection.prepare("UPDATE problems SET retrieved_at = ?1 WHERE retrieved_at IS NULL OR ( solution_data IS NULL AND retrieved_at < ?2 ) RETURNING problem_id, problem_data")?;
    let mut rows = statement.query(data)?;

    while let Some(row) = rows.next()? {
        problems.push(Problem {
            id: row.get(0)?,
            data: row.get(1)?,
        });
    }

    return Ok(problems);
}

fn update_solution(db_path: String, problem: Problem, solution_data: String) -> Result<()> {
    let connection = Connection::open(db_path)?;
    let updated_at = Local::now().to_rfc3339();
    let data = (&updated_at, &solution_data, &problem.id);
    connection.execute(
        "UPDATE problems SET updated_at = ?1, solution_data = ?2 WHERE problem_id = ?3",
        data,
    )?;
    return Ok(());
}

fn solve(db_path: String, problem: Problem) {
    let Ok(home_parameters) = serde_json::from_str(&problem.data) else { return };
    let home_problem = AdvancedHomeProblem::new(home_parameters);
    let solution = astar(&home_problem, |state| home_problem.heuristic_function(state), false);
    if solution.is_some() {
        println!("problem {:?} is solved", problem.id);
        let (plan, _) = solution.unwrap();
        let mut integer_plan = vec![];
        for action in plan {
            let mut appliance_actions = vec![];
            for appliance_action in action.appliances {
                appliance_actions.push(match appliance_action {
                    ApplianceAction::ON => 1,
                    ApplianceAction::OFF => 0,
                });
            }
            integer_plan.push(Action {
                battery: match action.battery {
                    BatteryAction::DISCHARGE => -1,
                    BatteryAction::OFF => 0,
                    BatteryAction::CHARGE => 1,
                },
                appliances: appliance_actions,
            });
        }
        let Ok(solution_data) = serde_json::to_string(&integer_plan) else { return };
        let Ok(_) = update_solution(db_path.clone(), problem, solution_data) else { return };
    } else {
        println!("problem {:?} is unsolvable", problem.id);
        let Ok(_) = update_solution(db_path.clone(), problem, "unsolvable".to_string()) else { return };
    }
}

fn main() {
    // basic_problem::run();
    // extended_problem::run();
    // advanced_problem::run();

    let started_at = Local::now();
    let pool = ThreadPool::new(4);

    loop {
        if let Ok(problems) = retrieve_problems("/Users/km17304/Workspace/cuttlefish/cuttlefish.db".to_string(), started_at) {
            for problem in problems {
                pool.execute(|| {
                    println!("adding problem {:?} to thread pool", problem.id);
                    solve("/Users/km17304/Workspace/cuttlefish/cuttlefish.db".to_string(), problem);
                });
            }
        }

        sleep(Duration::from_secs(1));
    }
}

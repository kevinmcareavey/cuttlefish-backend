use chrono::Local;
use rand::Rng;
use rusqlite::{Connection, Result, RowIndex};
use serde::{Deserialize, Serialize};
use crate::advanced_problem::AdvancedHomeProblem;
use crate::basic_problem::{ApplianceAction, BatteryAction, HomeProblem};
use crate::planner::astar;

mod data;
mod planner;
mod basic_problem;
mod extended_problem;
mod advanced_problem;

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

fn retrieve_problems(db_path: String) -> Result<Vec<Problem>> {
    let mut problems = vec![];

    let connection = Connection::open(db_path)?;
    let mut statement = connection.prepare("SELECT problem_id, problem_data FROM problems WHERE solution_data IS NULL LIMIT 1")?;
    let mut rows = statement.query([])?;

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
    println!("UPDATE {:?}", data);
    connection.execute(
        "UPDATE problems SET updated_at = ?1, solution_data = ?2 WHERE problem_id = ?3",
        data,
    )?;
    return Ok(());
}

fn run(db_path: String) {
    let Ok(problems) = retrieve_problems(db_path.clone()) else { return };

    for problem in problems {
        let Ok(home_parameters) = serde_json::from_str(&problem.data) else { continue };
        let home_problem = AdvancedHomeProblem::new(home_parameters);
        let solution = astar(&home_problem, |state| home_problem.heuristic_function(state), true);
        if solution.is_some() {
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
            let Ok(solution_data) = serde_json::to_string(&integer_plan) else { continue };
            let Ok(_) = update_solution(db_path.clone(), problem, solution_data) else { continue };
        } else {
            println!("{:?} is unsolvable", problem.id);
        }
    }
}

fn main() {
    // basic_problem::run();
    // extended_problem::run();
    // advanced_problem::run();
    run("/Users/kevin/Workspace/cuttlefish-api/shared.db".to_string());
}

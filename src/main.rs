use clap::Parser;

mod cli;
mod commands;

fn main() {
    cli::run(cli::Cli::parse());
}

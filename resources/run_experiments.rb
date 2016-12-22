#!/usr/bin/env ruby

require 'yaml'

# get filename from first command line argument or else resources/experiments.yaml
fname = ARGV.shift || 'resources/experiments.yaml'
yaml = YAML.load_file(fname)

# iterate through yaml - expected to be a list of hashes
yaml['experiments'].each.with_index do |experiment, i|
  repetitions = experiment.delete('repetitions')
  exec_args = experiment.keys.map { |key| '--' + key }.zip(experiment.values).join(' ')
  java_cmd = %{MAVEN_OPTS="-ea" mvn exec:java -Dexec.mainClass="es.valcarcelsainz.dce.DCEOptimizer" -Dexec.args="#{exec_args}"}

  repetitions.times.with_index do |j|
    puts "\nExperiment #{i+1}, repetition #{j+1}"
    puts java_cmd
    fork do
      sleep 15
      `redis-cli publish broadcast start`
    end
    system java_cmd
  end
end

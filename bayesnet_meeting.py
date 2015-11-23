from pomegranate import *

# Meeting Type
meeting_type = DiscreteDistribution({
	'breakfast': 1./2,
	'lunch': 1./2,
	# 'dinner': 1./6,
	# 'hangout': 1./6,
	# 'call': 1./6,
	# 'meeting': 1./6
})

duration = ConditionalProbabilityTable([
	['breakfast', 30, 1.0],
	['breakfast', 60, 0.0],
	['lunch', 30, 0.5],
	['lunch', 60, 0.5],
	# ['dinner', 30, 0.0],
	# ['dinner', 60, 1.0],
	# ['hangout', 30, 0.0],
	# ['hangout', 60, 1.0],
], [meeting_type])


# Make the states
s0 = State( meeting_type, name="type")
s1 = State( duration, name="duration")

# Make the bayes net, add the states, and the conditional dependencies.
network = BayesianNetwork( "meeting" )
network.add_states( [s0, s1] )
network.add_transition( s0, s1 )
network.bake()


# The first observation is that the duration is thirty minutes
first_observation = { 'duration' : 30 }

# beliefs will be an array of posterior distributions or clamped values for each state, indexed corresponding to the order
# in self.states.
#beliefs = network.forward_backward( first_observation )

# Convert the beliefs into a more readable format
#beliefs = map( str, beliefs )

# Print out the state name and belief for each state on individual lines
#print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )


### Testing the new add_items method

print "== Meeting Type BEFORE =="
print meeting_type

print "== Add new key to meeting type =="
meeting_type.add_items( ['call'] )
meeting_type.train( ['call'], inertia=0.7 )

print "== Meeting Type AFTER =="
print meeting_type

print

print "== Duration BEFORE =="
print duration

print "== Add new row to the table =="
# I have to add every new possibility manually
duration.add_items( [['call', 30], ['call',60],['call', 120],['breakfast', 120], ['lunch', 120]] )
duration.train( [['call', 120], ['call', 30]] )

print "== Duration AFTER =="
print duration

print

print "== Network Training =="
data = [ ['call', 30], ['breakfast', 120], ['lunch', 120] ]
network.train( data )
print "== Meeting Type =="
print meeting_type
print "== Duration =="
print duration

network.bake()

# The second observation is that the meeting type is a 'meeting'
# second_observation = { 'type' : 'call' } # Works
second_observation = { 'duration' : 120 }

# beliefs will be an array of posterior distributions or clamped values for each state, indexed corresponding to the order
# in self.states.
beliefs = network.forward_backward( second_observation )

# Convert the beliefs into a more readable format
beliefs = map( str, beliefs )

# Print out the state name and belief for each state on individual lines
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )

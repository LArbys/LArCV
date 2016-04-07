class Storage(object) :
    def __init__(self,name='storage'):
        self.name = name

        ## Types of particles
        self.particle_types  = [ 
            "Eminus",
            "Kminus",
            "Proton",
            "Muminus",
            "Piminus",
            "Gamma",
            "Pizero",
            "BNB",
            "Cosmic",
            "Unknown"
        ]


        ### Plane colors
        self.colors = { 0 : 'r', 1 : 'g' , 2 : 'b' }

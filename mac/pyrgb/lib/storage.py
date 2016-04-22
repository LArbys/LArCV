class Storage(object) :
    def __init__(self,name='storage'):
        self.name = name

        ## Types of particles
        self.particle_types  = [
            "Unknown",
            "Cosmic",
            "BNB",
            "Eminus",
            "Gamma",
            "Pizero",
            "Muminus",
            "Kminus",
            "Piminus",
            "Proton"
        ]


        ### Plane colors
        self.colors = { 0 : 'r', 1 : 'g' , 2 : 'b' }

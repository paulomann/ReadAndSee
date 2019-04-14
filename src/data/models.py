class Questionnaire:
    """ Model that takes all information from a participant answer to the
    research questionnaire
    """
    def __init__(self, answer_dict):
        """ Every date attribute is according to this pattern: YYY-MM-DD """
        self.form_application_date = answer_dict["form_application_date"]
        self.email = answer_dict["email"]
        self.sex = answer_dict["sex"]
        self.birth_date = answer_dict["birth_date"]
        self.household_income = answer_dict["household_income"]
        self.facebook_hours = answer_dict["facebook_hours"]
        self.twitter_hours = answer_dict["twitter_hours"]
        self.instagram_hours = answer_dict["instagram_hours"]
        self.academic_degree = answer_dict["academic_degree"]
        self.course_name = answer_dict["course_name"]
        self.semesters = answer_dict["semesters"]
        self.scholarship = answer_dict["scholarship"]
        self.accommodation = answer_dict["accommodation"]
        self.works = answer_dict["works"]
        self.depression_diagnosed = answer_dict["depression_diagnosed"]
        self.in_therapy = answer_dict["in_therapy"]
        self.antidepressants = answer_dict["antidepressants"]
        self.twitter_user_name = answer_dict["twitter_user_name"]
        self.instagram_user_name = answer_dict["instagram_user_name"]
        self.BDI = answer_dict["BDI"]
        self.answer_dict = answer_dict

    def get_bdi_sum(self):
        return self.BDI

    def get_bdi_category(self):
        """ Return the bdi category as a discrete category value.

        Return:
        0 -- for BDI less than 14 (minimal)
        1 -- for BDI greater or equal than 14 and less than 20 (mild)
        2 -- for BDI greater or equal than 20 and less than 29 (moderate)
        3 -- for BDI greater or equal 29 (severe)
        """
        n = self.BDI
        if n < 14:
            return 0
        elif n >= 14 and n < 20:
            return 1
        elif n >= 20 and n < 29:
            return 2
        else:
            return 3


class Participant:
    """ The generalization of a participant of our study.

    Since every participant answer the questionnaire, has a username and a list
    of posts, this is the most general case for instagram or twitter users.
    """
    def __init__(self, questionnaire, username, posts):
        self.questionnaire = questionnaire
        self.username = username
        self.posts = posts

    def get_answer_dict(self):
        return self.questionnaire.answer_dict


class InstagramUser(Participant):

    """
    Instagram's profile information data model, with the biodemographic
    information
    """
    def __init__(self, biography, followers_count, following_count,
                 is_private, posts_count, questionnaire, username,
                 instagram_posts):
        """
        Params:
        questionnaire -- It contains the answer to the questionnaire from this
            InstagramUser, which is a Questionnaire object.
        """
        super().__init__(questionnaire, username, instagram_posts)
        self.biography = biography
        self.followers_count = followers_count
        self.following_count = following_count
        self.is_private = is_private
        self.posts_count = posts_count

    def get_answer_dict(self):
        """ Return the questionnaire dict with appended Instagram data.

        Also return the keys of the new appended dict.

        Return:
        qtnre -- New questionnaire dict, with appended data
        cols -- Appended Data keys
        """
        instagram_user_data = dict(followers_count=self.followers_count,
                                   following_count=self.following_count,
                                   posts_count=self.posts_count)
        qtnre = super().get_answer_dict().copy()
        qtnre.update(instagram_user_data)
        return qtnre, list(instagram_user_data.keys())


class InstagramPost:

    """
    This class models all attributes gathered from the Instagram scraped data.

    Represents an abstraction for the instagram post entity with all its fields

    """

    def __init__(self, img_path_list, caption, likes_count, timestamp,
                 comments_count):
        """
        Params:
        instagram_user -- Object from InstagramUser class.
        img_paths -- A list of image paths related to a post
        """
        self._img_path_list = img_path_list
        # self.pics_ids = None # TODO: Derivated attribute from img_paths
        self.caption = caption
        self.likes_count = likes_count
        self.timestamp = timestamp
        self.comments_count = comments_count
        self._face_count_list = []
        # self.face_presence = True if face_count > 0 else False

    def get_img_path_list(self):
        if len(self._face_count_list) == len(self._img_path_list):
            return self._img_path_list
        else:
            return None

    def get_face_count_list(self):
        if len(self._face_count_list) == len(self._img_path_list):
            return self._face_count_list
        else:
            return None

    def calculate_face_count_list(self):
        """ Calculate face count for each picture in the post.

        Return:
        True -- If the face_count_list is the same size as the img_path_list
        False -- Otherwise
        """
        self.face_count_list = []
        for img_path in self.img_paths_list:
            img = self._load_image(img_path)
            face_count = self._get_face_count(img)
            self.face_count_list.append(face_count)
        if len(self.face_count_list) == len(self.img_path_list):
            return True
        return False

    def _load_image(self, image_path):
        return face_recognition.load_image_file(image_path)

    def _get_face_count(self, image):
        face_locations = face_recognition.face_locations(image)
        return len(face_locations)

class InstagramProfile:

    """
    Instagram's profile information data model
    """
    def __init__(self, biography, followers_count, following_count,
                 is_private, posts_count):
        self.biography = biography
        self.followers_count = followers_count
        self.following_count = following_count
        self.is_private = is_private
        self.posts_count = posts_count


class InstagramPost:

    """
    This class models all attributes gathered from the Instagram scraped data.

    Represents an abstraction for the instagram post entity with all its fields.

    """

    def __init__(self, img_path, caption, like_count, timestamp,
                 comments_count, pic_id, instagram_profile, face_count):
        self.img_path = img_path
        self.caption = caption
        self.like_count = like_count
        self.timestamp = timestamp
        self.comments_count = comments_count
        self.pic_id = pic_id
        self.instagram_profile = instagram_profile
        self.face_count = face_count
        self.face_presence = True if face_count > 0 else False

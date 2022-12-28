from abc import abstractmethod

from preprocessing_pgp.phone.const import (
    SUB_PHONE_10NUM,
    SUB_PHONE_11NUM,
    SUB_TELEPHONE_10NUM,
    SUB_TELEPHONE_11NUM,
    PHONE_LENGTH
)


class PhoneValidator:
    """
    Abstract Class contains validating functions for verify phone
    """

    @abstractmethod
    def is_valid_phone(self, phone: str) -> bool:
        """
        Check whether the phone is valid

        Parameters
        ----------
        phone : str
            The input phone string to check for validation

        Returns
        -------
        bool
            Whether it is valid or not

        Raises
        ------
        NotImplementedError
            The subclass not override this function
        """
        raise NotImplementedError("Subclasses should implement this function!")


class MobiPhoneValidator(PhoneValidator):
    """
    Class to check for mobi phone syntax
    """

    def __init__(self):
        self.old_length = PHONE_LENGTH['old_mobi']
        self.new_length = PHONE_LENGTH['new_mobi']

    def _is_new_sub_phone(self, phone: str) -> bool:
        return phone[:3] in SUB_PHONE_10NUM

    def _is_old_sub_phone(self, phone: str) -> bool:
        return phone[:3] in SUB_PHONE_11NUM

    def _is_new_phone_length(self, phone: str) -> bool:
        return len(phone) == self.new_length

    def _is_old_phone_length(self, phone: str) -> bool:
        return len(phone) == self.old_length

    def is_new_phone(self, phone: str) -> bool:
        """
        Check whether the phone is `new mobi` phone

        Parameters
        ----------
        phone : str
            The input phone number sequence

        Returns
        -------
        bool
            Whether the phone is `new mobi` phone
        """

        return self._is_new_phone_length(phone)\
            and self._is_new_sub_phone(phone)

    def is_old_phone(self, phone: str) -> bool:
        """
        Check whether the phone is `old mobi` phone

        Parameters
        ----------
        phone : str
            The input phone number sequence

        Returns
        -------
        bool
            Whether the phone is `old mobi` phone
        """

        return self._is_old_phone_length(phone)\
            and self._is_old_sub_phone(phone)

    def is_valid_phone(self, phone: str) -> bool:
        """
        General function to check whether the phone is valid phone number or not

        Parameters
        ----------
        phone : str
            The input phone number sequence

        Returns
        -------
        bool
            Whether the phone is valid phone
        """
        return self.is_new_phone(phone)\
            or self.is_old_phone(phone)


class LandlinePhoneValidator(PhoneValidator):
    """
    Class to check for mobi phone syntax
    """

    def __init__(self):
        self.old_length = PHONE_LENGTH['old_landline']
        self.new_length = PHONE_LENGTH['new_landline']

    def _is_new_sub_phone(self, phone: str) -> bool:
        return phone[:3] in SUB_TELEPHONE_11NUM\
            or phone[:4] in SUB_TELEPHONE_11NUM

    def _is_old_sub_phone(self, phone: str) -> bool:
        return phone[:2] in SUB_TELEPHONE_10NUM\
            or phone[:3] in SUB_TELEPHONE_10NUM\
            or phone[:4] in SUB_TELEPHONE_10NUM

    def _is_new_phone_length(self, phone: str) -> bool:
        return len(phone) == self.new_length

    def _is_old_phone_length(self, phone: str) -> bool:
        return len(phone) == self.old_length

    def is_new_phone(self, phone: str) -> bool:
        """
        Check whether the phone is `new mobi` phone

        Parameters
        ----------
        phone : str
            The input phone number sequence

        Returns
        -------
        bool
            Whether the phone is `new mobi` phone
        """

        return self._is_new_phone_length(phone)\
            and self._is_new_sub_phone(phone)

    def is_old_phone(self, phone: str) -> bool:
        """
        Check whether the phone is `old mobi` phone

        Parameters
        ----------
        phone : str
            The input phone number sequence

        Returns
        -------
        bool
            Whether the phone is `old mobi` phone
        """

        return self._is_old_phone_length(phone)\
            and self._is_old_sub_phone(phone)

    def is_valid_phone(self, phone: str) -> bool:
        """
        General function to check whether the phone is valid phone number or not

        Parameters
        ----------
        phone : str
            The input phone number sequence

        Returns
        -------
        bool
            Whether the phone is valid phone
        """
        return self.is_new_phone(phone)\
            or self.is_old_phone(phone)

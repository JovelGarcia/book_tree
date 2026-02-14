import { Routes } from '@angular/router';
import { EpubListComponent } from './pages/epub-list/epub-list.component';
import { EpubDetailComponent } from './pages/epub-detail/epub-detail.component';
import { UploadComponent } from './pages/upload/upload.component';
import { ConfirmDeleteComponent } from './pages/confirm-delete/confirm-delete.component';

export const routes: Routes = [
  { path: '', component: EpubListComponent },
  { path: 'epub/:id', component: EpubDetailComponent },
  { path: 'upload', component: UploadComponent },
  { path: 'delete/:id', component: ConfirmDeleteComponent },
];
